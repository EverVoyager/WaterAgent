"""LangGraph 图节点函数。

从 workflow.py 拆分而来。依赖 state, errors, cache, llm_helpers, synthesizer_node,
agent.prompts, agent.router, app.core.llm。
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import (
    APIConnectionError,
    APITimeoutError,
    APIError,
    RateLimitError,
)

from agent.graph.cache import _cached_execute_tool
from agent.graph.errors import _classify_llm_error
from agent.graph.state import AgentState
from agent.graph.synthesizer_node import _summarize_results
from agent.prompts import DIRECT_CHAT_PROMPT
from agent.router import detect_intent
from app.core.llm import LLM_TIMEOUTS, get_llm_client, get_llm_config, strip_think

logger = logging.getLogger(__name__)

MAX_ROUNDS = 3  # 规划最大循环次数（原 reflector 硬限制）

# 工具执行线程池（P3 并发执行）
_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool-exec")


# ====== 节点函数 ======

def router_node(state: AgentState) -> Dict[str, Any]:
    """路由节点：识别意图，初始化状态。

    使用 Semantic Router（embedding 余弦相似度）做主路径意图识别，
    embedding 不可用时退化为规则化兜底。
    """
    query = state["user_query"]
    intent, decision = detect_intent(query)
    logger.info(
        "[router] query=%s intent=%s score=%.3f fallback=%s",
        query[:60], intent, decision.score, decision.fallback_reason or "-",
    )
    return {
        "intent": intent,
        "rounds": 0,
        "tool_results": {},
        "tool_calls": [],
    }


def direct_chat_node(state: AgentState) -> Dict[str, Any]:
    """闲聊节点：直接调用 LLM 对话，跳过工具与研判。LLM 不可用时直接抛 LLMError。"""
    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["chat"])
    query = state["user_query"]
    history = state.get("history", [])

    messages = [{"role": "system", "content": DIRECT_CHAT_PROMPT}]
    for m in history[-6:]:  # 仅保留最近 3 轮
        messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    messages.append({"role": "user", "content": query})

    try:
        resp = client.chat.completions.create(
            model=settings["model"],
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
    except (APITimeoutError, RateLimitError, APIConnectionError, APIError) as e:
        logger.exception("[direct_chat] LLM 调用失败 (%s)", type(e).__name__)
        raise _classify_llm_error(e) from e
    except Exception as e:
        logger.exception("[direct_chat] LLM 未知异常")
        raise _classify_llm_error(e) from e

    answer = strip_think((resp.choices[0].message.content or "").strip())
    logger.info("[direct_chat] answer=%s", answer[:80])
    return {
        "final_answer": answer,
        "warning_level": "",     # 闲聊不输出预警等级
        "reasoning": "",
        "actions": [],
        "tool_calls": [],
    }


def _direct_chat_stream(query: str, history: List[Dict[str, Any]]):
    """流式版本的闲聊生成器。使用 LLM stream=True，逐 token yield answer_delta。

    借鉴 LangChain astream_events 的 on_chat_model_stream 事件思路。
    LLM 调用失败时抛 LLMError。
    """
    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["chat"])

    messages = [{"role": "system", "content": DIRECT_CHAT_PROMPT}]
    for m in history[-6:]:
        messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    messages.append({"role": "user", "content": query})

    try:
        stream = client.chat.completions.create(
            model=settings["model"],
            messages=messages,
            temperature=0.7,
            max_tokens=512,
            stream=True,
        )
    except (APITimeoutError, RateLimitError, APIConnectionError, APIError) as e:
        logger.exception("[direct_chat_stream] LLM 流式调用失败 (%s)", type(e).__name__)
        raise _classify_llm_error(e) from e
    except Exception as e:
        logger.exception("[direct_chat_stream] LLM 未知异常")
        raise _classify_llm_error(e) from e

    # Qwen3 思考内容剥离：流式需缓冲检测 <think>...</think> 块
    # 与 llm_helpers._stream_llm 保持一致的状态机逻辑
    filtered_answer = []
    buffer = ""
    in_think = False
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if not content:
            continue
        buffer += content
        output = ""
        while buffer:
            if in_think:
                end = buffer.find("</think>")
                if end == -1:
                    # think 块未结束，继续缓冲
                    break
                # 跳过 think 内容及结束标签
                buffer = buffer[end + len("</think>"):]
                while buffer and buffer[0] in " \n\t":
                    buffer = buffer[1:]
                in_think = False
            else:
                start = buffer.find("<think>")
                if start == -1:
                    # 没有 think 标签，但保留末尾可能是 "<think" 前缀的部分
                    partial = 0
                    for plen in range(min(len(buffer), len("<think>")), 0, -1):
                        if buffer.endswith("<think>"[:plen]):
                            partial = plen
                            break
                    output += buffer[:len(buffer) - partial]
                    buffer = buffer[len(buffer) - partial:]
                    break
                output += buffer[:start]
                buffer = buffer[start + len("<think>"):]
                in_think = True
        if output:
            filtered_answer.append(output)
            yield {"type": "answer_delta", "content": output}

    # 流结束时若 think 块仍未闭合（异常中断），丢弃残留 buffer
    # 推送完整 answer 供 done 事件使用（已过滤 think）
    yield {"type": "synth_answer_full", "content": "".join(filtered_answer)}


def planner_node(state: AgentState) -> Dict[str, Any]:
    """规划节点（P4 合并 reflector）：使用 LLM 原生 Function Calling 决策工具调用。

    同时承担原 reflector 的"信息是否充分"判断职责：
    - LLM 不返回 tool_calls → should_continue=False（信息已充分，进入 synthesizer）
    - LLM 返回 tool_calls → should_continue=True（继续执行工具）
    - 达到 MAX_ROUNDS → should_continue=False（强制结束避免死循环）

    LLM 调用失败时抛 LLMError。
    """
    rounds = state.get("rounds", 0) + 1
    query = state["user_query"]
    context_summary = _summarize_results(state.get("tool_results", {}))
    # 已调用过的工具列表（去重签名：name+关键参数），传给 LLM 避免重复决策
    called_tools = _summarize_called_tools(state.get("tool_calls", []))

    # 自进化：第 1 轮规划时注入历史经验（成功工具模式 + 失败教训）
    experiences = ""
    if rounds == 1:
        try:
            from agent.memory import get_relevant_experiences
            experiences = get_relevant_experiences(query)
            if experiences:
                logger.info("[planner] 注入历史经验：\n%s", experiences[:200])
        except Exception as e:
            logger.debug("[planner] 注入经验失败（不影响主流程）：%s", e)

    planned = _plan_via_function_calling(query, context_summary, called_tools, rounds, experiences)

    # 去重：如果 LLM 返回的工具调用与历史完全相同（name+arguments），跳过避免死循环
    planned = _dedupe_planned_calls(planned, state.get("tool_calls", []))

    # P4 合并 reflector：planner 自行判断是否继续
    # 规则：planned 为空 → 信息已充分；达到 MAX_ROUNDS → 强制结束
    if rounds >= MAX_ROUNDS:
        should_continue = False
        logger.info("[planner] round=%d MAX_ROUNDS reached, forcing stop", rounds)
    elif not planned:
        should_continue = False
        logger.info("[planner] round=%d no tool_calls, info sufficient", rounds)
    else:
        should_continue = True

    logger.info("[planner] round=%d planned=%s should_continue=%s",
                rounds, planned, should_continue)
    return {"rounds": rounds, "planned_calls": planned, "should_continue": should_continue}


def _summarize_called_tools(tool_calls: List[Dict[str, Any]]) -> str:
    """把已调用的工具列表格式化为 LLM 可读的摘要。"""
    if not tool_calls:
        return "(暂无)"
    seen = set()
    parts = []
    for tc in tool_calls:
        name = tc.get("tool_name", "")
        args = tc.get("arguments", {})
        # 关键参数摘要
        key_args = {k: v for k, v in args.items() if k in ("station", "location", "metric", "lead_time_hours")}
        sig = f"{name}({key_args})"
        if sig in seen:
            continue
        seen.add(sig)
        parts.append(sig)
    return "、".join(parts) if parts else "(暂无)"


def _dedupe_planned_calls(
    planned: List[Dict[str, Any]],
    history_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """去除与历史完全相同（name+arguments）的重复调用，避免死循环。"""
    if not planned or not history_calls:
        return planned
    history_sigs = set()
    for tc in history_calls:
        name = tc.get("tool_name", "")
        args = tc.get("arguments", {})
        try:
            sig = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
        except (TypeError, ValueError):
            sig = f"{name}:{args}"
        history_sigs.add(sig)
    result = []
    for call in planned:
        name = call.get("name", "")
        args = call.get("arguments", {}) or {}
        try:
            sig = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
        except (TypeError, ValueError):
            sig = f"{name}:{args}"
        if sig in history_sigs:
            logger.info("[planner] skip duplicate call: %s args=%s", name, args)
            continue
        result.append(call)
    return result


def _plan_via_function_calling(
    query: str,
    context_summary: str,
    called_tools: str = "",
    round_num: int = 1,
    experiences: str = "",
) -> List[Dict[str, Any]]:
    """通过 LLM 原生 Function Calling 规划工具调用。

    让模型自主决定调用哪些工具及参数。LLM 调用失败时抛 LLMError。
    planner 同时承担"信息是否充分"判断：返回空 tool_calls 表示信息已充分。
    """
    from agent.tools.schemas import build_openai_tools

    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["planner"])
    tools_schema = build_openai_tools()

    system_prompt = (
        "你是黄河吕梁段防汛预警智能体的工具调用规划模块。"
        "根据用户问题和已收集的信息，决定本轮需要调用哪些工具。"
        "可以一次调用多个工具，也可以不调用（如果信息已充分）。\n"
        "重要：\n"
        "1. 如果已收集的信息已足够回答用户问题，请返回空工具调用列表。\n"
        "2. 避免重复调用已调用过的工具（除非参数明显不同需要重新查询）。\n"
        "3. 第 1 轮若需要工具，优先调用最关键的 1-3 个。\n"
        "4. 涉及实时数据（天气/水情/径流/地形/法规）的查询，必须调用对应工具获取真实数据，"
        "严禁凭自身知识回答（自身知识可能过时或不准确）。\n"
        "5. 只有纯常识性闲聊问题（如'什么是防汛'）才可不调用工具。\n"
    )
    # 自进化：注入历史经验时附加指导
    exp_section = ""
    if experiences:
        exp_section = (
            f"\n\n历史经验（参考，可借鉴但不必完全照搬）：\n{experiences}\n\n"
            "提示：若历史经验中的工具组合适用于当前问题，可优先采用。"
        )

    user_prompt = (
        f"用户问题：{query}\n\n"
        f"已收集信息：{context_summary}\n\n"
        f"已调用过的工具：{called_tools}{exp_section}\n\n"
        f"当前是第 {round_num} 轮规划。请决定本轮需要调用的工具。"
        f"若信息已充分，请不调用任何工具。"
    )

    try:
        resp = client.chat.completions.create(
            model=settings["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=tools_schema,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=1024,
        )
    except (APITimeoutError, RateLimitError, APIConnectionError, APIError) as e:
        logger.exception("[planner] LLM Function Calling 调用失败 (%s)", type(e).__name__)
        raise _classify_llm_error(e) from e
    except Exception as e:
        logger.exception("[planner] LLM 未知异常")
        raise _classify_llm_error(e) from e

    msg = resp.choices[0].message
    if not msg.tool_calls:
        logger.info("[planner] LLM decided no tool calls needed (info sufficient)")
        return []

    planned = []
    for tc in msg.tool_calls:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            planned.append({"name": tc.function.name, "arguments": args})
        except json.JSONDecodeError as e:
            logger.warning("[planner] failed to parse args for %s: %s", tc.function.name, e)
            continue
    return planned


def _execute_one_tool(
    call: Dict[str, Any],
    idx: int,
    weather_rainfall_mm: float,
    weather_series: Optional[List[dict]],
    round_num: int,
    existing_keys: set,
) -> Dict[str, Any]:
    """执行单个工具调用，返回结构化记录。

    封装为独立函数以支持 P3 并发执行。包含 P6 缓存命中、跨工具数据流注入。
    """
    name = call.get("name", "")
    args = dict(call.get("arguments", {}) or {})
    error = ""
    result: Dict[str, Any] = {}

    if not name:
        return {"name": name, "args": args, "result": result, "error": "empty tool name",
                "result_key": f"empty_{idx}", "is_weather": False}

    # 跨工具数据流：predict_runoff 自动注入降雨量
    if name == "predict_runoff" and weather_rainfall_mm > 0:
        args.setdefault("rainfall_mm", weather_rainfall_mm)
        if weather_series:
            args.setdefault("rainfall_series", weather_series)
        logger.info(
            "[executor] predict_runoff 注入降雨数据: rainfall_mm=%.1f",
            weather_rainfall_mm,
        )

    logger.info("[executor] call %s args=%s", name, args)
    try:
        result = _cached_execute_tool(name, args)  # P6 缓存
    except Exception as e:
        logger.exception("[executor] tool failed: %s", name)
        result = {}
        error = str(e)

    # 累积结果（同名工具用 idx 区分）
    result_key = f"{name}_{idx}" if name in existing_keys else name

    return {
        "name": name,
        "args": args,
        "result": result,
        "error": error,
        "result_key": result_key,
        "is_weather": name == "get_weather" and isinstance(result, dict),
    }


def executor_node(state: AgentState) -> Dict[str, Any]:
    """执行节点（P3 并发 + P6 缓存）：并行执行所有计划工具。

    支持跨工具数据流：若本轮或历史已调用 get_weather，调用 predict_runoff 时
    自动注入 rainfall_mm + rainfall_series。

    并发策略：
    - 无 predict_runoff 时，所有工具并发
    - 有 predict_runoff 时，先并发执行 get_weather，再并发执行其余（保证注入）
    """
    planned = state.get("planned_calls", [])
    tool_results = dict(state.get("tool_results", {}))
    tool_calls = list(state.get("tool_calls", []))
    round_num = state.get("rounds", 1)
    existing_keys = set(tool_results.keys())

    if not planned:
        return {"tool_results": tool_results, "tool_calls": tool_calls}

    # 查找历史 get_weather 结果（用于跨工具数据流）
    weather_rainfall_mm: float = 0.0
    weather_series: Optional[List[dict]] = None
    for k, v in tool_results.items():
        if k.startswith("get_weather") and isinstance(v, dict):
            if v.get("total_rainfall_mm"):
                weather_rainfall_mm = float(v["total_rainfall_mm"])
            if v.get("series"):
                weather_series = v["series"]
            break

    # 检查本轮是否同时包含 get_weather 和 predict_runoff
    has_weather = any(c.get("name") == "get_weather" for c in planned if isinstance(c, dict))
    has_runoff = any(c.get("name") == "predict_runoff" for c in planned if isinstance(c, dict))

    # 分阶段执行：阶段1跑 weather（拿到降雨数据），阶段2跑其余（含注入后的 runoff）
    if has_weather and has_runoff:
        stage1 = [(i, c) for i, c in enumerate(planned)
                  if isinstance(c, dict) and c.get("name") == "get_weather"]
        stage2 = [(i, c) for i, c in enumerate(planned)
                  if isinstance(c, dict) and c.get("name") != "get_weather"]
    else:
        stage1 = []
        stage2 = [(i, c) for i, c in enumerate(planned) if isinstance(c, dict)]

    def _run_stage(stage_items):
        if not stage_items:
            return []
        # 单工具直接串行，避免线程池开销
        if len(stage_items) == 1:
            i, c = stage_items[0]
            return [_execute_one_tool(c, i, weather_rainfall_mm, weather_series,
                                       round_num, existing_keys)]
        # 多工具并发
        futures = []
        for i, c in stage_items:
            fut = _TOOL_EXECUTOR.submit(
                _execute_one_tool, c, i, weather_rainfall_mm, weather_series,
                round_num, existing_keys,
            )
            futures.append(fut)
        results = []
        for fut in as_completed(futures):
            results.append(fut.result())
        # 按 idx 排序保持顺序
        results.sort(key=lambda r: planned.index(next(c for i, c in stage_items
                                                       if r["name"] == c.get("name", ""))))
        return results

    # 阶段 1：执行 weather（如果有的话）
    stage1_results = _run_stage(stage1)
    for r in stage1_results:
        tool_results[r["result_key"]] = r["result"]
        tool_calls.append({
            "tool_name": r["name"],
            "arguments": r["args"],
            "result": r["result"],
            "error": r["error"],
            "round": round_num,
        })
        # 同轮 get_weather 结果立即更新，供 predict_runoff 使用
        if r["is_weather"]:
            if r["result"].get("total_rainfall_mm"):
                weather_rainfall_mm = float(r["result"]["total_rainfall_mm"])
            if r["result"].get("series"):
                weather_series = r["result"]["series"]
        existing_keys.add(r["result_key"])

    # 阶段 2：执行其余工具（含注入后的 predict_runoff）
    stage2_results = _run_stage(stage2)
    for r in stage2_results:
        tool_results[r["result_key"]] = r["result"]
        tool_calls.append({
            "tool_name": r["name"],
            "arguments": r["args"],
            "result": r["result"],
            "error": r["error"],
            "round": round_num,
        })

    return {"tool_results": tool_results, "tool_calls": tool_calls}


def reflector_node(state: AgentState) -> Dict[str, Any]:
    """反思节点（P4 已废弃，保留空实现向后兼容）。

    原 reflector 职责已合并到 planner_node。此函数仅为兼容旧图定义保留，
    不应被新图调用。新图直接基于 planner 的 should_continue 决策路由。
    """
    raise RuntimeError(
        "reflector_node 已废弃（P4 合并到 planner_node），请检查图定义"
    )
