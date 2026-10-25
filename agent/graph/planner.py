"""工具规划节点：原生 FC 轨迹、调用去重与完成度守卫。"""
import json
import logging
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)

from agent.graph.errors import _classify_llm_error
from agent.graph.planner_context import _build_fc_round1_messages, _build_status_bar, get_max_rounds
from agent.graph.state import AgentState
from app.core.llm import LLM_TIMEOUTS, extract_content, get_llm_client, get_llm_config
from app.core.llm_stats import record_llm_usage

logger = logging.getLogger(__name__)


def planner_node(state: AgentState) -> dict[str, Any]:
    """规划节点（P4 合并 reflector）：LLM 原生 Function Calling 消息序列驱动。

    fc_messages 为请求内累积的原生消息序列（user → assistant(tool_calls) →
    tool(结果) → …），只追加不重写——模型在其训练分布内看到自己的决策轨迹
    与工具原始返回，而非每轮重建的文本摘要（对齐 Claude Code / Codex 的
    agent loop 形态；KV Cache 前缀只增不改）。

    - 第 1 轮构建 [system, user(问题+上下文工程段)]；后续轮续用序列并在
      末尾追加状态提示（system-reminder 式：时间/进度/收尾指令）
    - 模型返回的 assistant 消息原样追加（含 reasoning_content——DeepSeek
      思考模式续轮强制要求回传；其他后端不返回该字段则不携带）
    - 守卫补充的调用合成 assistant(tool_calls)（占位 reasoning_content），
      保证每个 tool_call 都有配对的 tool 消息（API 配对约束）

    同时承担原 reflector 的"信息是否充分"判断职责：
    - LLM 不返回 tool_calls → should_continue=False（信息已充分，进入 synthesizer）
    - LLM 返回 tool_calls → should_continue=True（继续执行工具）
    - 达到 LLM_MAX_TOOL_ROUNDS → should_continue=False（强制结束避免死循环）

    Skill 机制（借鉴 Claude Skills）：第 1 轮匹配 Skill，注入行为指令 + 工具子集过滤。
    LLM 调用失败时抛 LLMError。
    """
    rounds = state.get("rounds", 0) + 1
    query = state["user_query"]

    # Skill 匹配（借鉴 Claude Skills 按需加载）：第 1 轮匹配，后续轮次复用 state 中的结果
    skill_instructions = state.get("skill_instructions", "")
    skill_tool_names = state.get("skill_tool_names", [])
    skill_name = state.get("skill_name", "")
    if rounds == 1 and not skill_instructions:
        try:
            from agent.skills import match_skill
            matched = match_skill(query)
            if matched:
                skill_name = matched.name
                skill_instructions = matched.instructions
                skill_tool_names = matched.tool_names
                logger.info(
                    "[planner] 匹配到 Skill: %s (tools=%s)",
                    skill_name, skill_tool_names or "all",
                )
        except Exception as e:
            logger.debug("[planner] Skill 匹配失败（不影响主流程）：%s", e)

    # 自进化：第 1 轮规划时注入历史经验（成功工具模式 + 失败教训）。
    # 结果写入 state 跨轮原样保留（KV Cache 前缀"只增不改"）
    experiences = state.get("experiences", "")
    if rounds == 1 and not experiences:
        try:
            from agent.memory import get_relevant_experiences
            experiences = get_relevant_experiences(query)
            if experiences:
                logger.info("[planner] 注入历史经验：\n%s", experiences[:200])
        except Exception as e:
            logger.debug("[planner] 注入经验失败（不影响主流程）：%s", e)

    # 上下文压缩：第 1 轮注入历史对话摘要（含压缩后的早轮摘要 + 最近几轮原文）
    history_context = state.get("history_context", "")
    if rounds == 1 and not history_context:
        history = state.get("history", [])
        if history:
            try:
                from agent.graph.context_compact import extract_history_context
                history_context = extract_history_context(history)
                if history_context:
                    logger.info("[planner] 注入历史摘要：\n%s", history_context[:200])
            except Exception as e:
                logger.debug("[planner] 注入历史摘要失败（不影响主流程）：%s", e)

    # 构建本轮原生消息序列
    if rounds == 1:
        fc_messages = _build_fc_round1_messages(
            query,
            skill_instructions=skill_instructions,
            experiences=experiences,
            history_context=history_context,
            recalled_context=state.get("recalled_context", ""),
        )
    else:
        fc_messages = list(state.get("fc_messages", []))
        # 末尾追加状态提示（时间/进度/收尾指令），system-reminder 式
        fc_messages.append({
            "role": "user",
            "content": _build_status_bar(rounds) + "\n请决定下一轮需要调用的工具；若信息已充分，请不调用任何工具。",
        })

    planned, assistant_msg = _plan_via_fc(fc_messages, skill_tool_names)

    # 去重：如果 LLM 返回的工具调用与历史完全相同（name+arguments），跳过避免死循环
    planned = _dedupe_planned_calls(planned, state.get("tool_calls", []))

    # 守卫 1：声称核验闸（反讨好）——用户口头声称预警等级但未规划数据工具时，
    # 强制追加数据核验（评估失效模式 2：陷阱抵抗 33%）。数据进场后由
    # synthesizer 的等级一致性门锚定规则引擎真值。
    if rounds == 1:
        from agent.graph.planner_guard import enforce_claim_verification
        planned = enforce_claim_verification(query, planned)

    # P4 合并 reflector：planner 自行判断是否继续
    # 规则：planned 为空 → 信息已充分；达到 max_rounds → 强制结束
    max_rounds = get_max_rounds()
    if rounds >= max_rounds:
        should_continue = False
        logger.info("[planner] round=%d max_rounds=%d reached, forcing stop", rounds, max_rounds)
    elif not planned:
        # 守卫 2：工具完成度检查——预案/研判类查询的关键工具缺失时不放行，
        # 强制补充一轮（评估失效模式 3：工具召回 48.1%）
        from agent.graph.planner_guard import missing_required_tools
        called_names = {tc.get("tool_name") for tc in state.get("tool_calls", [])}
        completion_calls = missing_required_tools(
            query, called_names, state.get("tool_results", {}),
        )
        if completion_calls:
            planned = completion_calls
            should_continue = True
            logger.info("[planner] round=%d 完成度闸补充工具: %s",
                        rounds, [c["name"] for c in planned])
        else:
            should_continue = False
            logger.info("[planner] round=%d no tool_calls, info sufficient", rounds)
    else:
        should_continue = True

    # 系统补充的调用（守卫/完成度闸）：分配合成 id，合并进本轮 assistant 消息
    # （占位 reasoning_content——DeepSeek 思考模式要求 assistant 消息携带该字段，
    # 实测占位文本可通过校验）
    synth_calls = [c for c in planned if not c.get("id")]
    for i, c in enumerate(synth_calls):
        c["id"] = f"call_sys_{rounds}_{i}"

    def _synth_tc(call: dict[str, Any]) -> dict[str, Any]:
        return {"id": call["id"], "type": "function",
                "function": {"name": call["name"],
                             "arguments": json.dumps(
                                 call.get("arguments", {}), ensure_ascii=False)}}

    # 追加本轮 assistant 消息（维持 tool_call/tool 配对约束）：
    # - 守卫补充的调用必须合并在同一条 assistant 消息里——若另起一条
    #   assistant(tool_calls)，两条 assistant 之间没有 tool 消息，
    #   违反 API 配对约束（DeepSeek/OpenAI 400：insufficient tool messages，
    #   2026-09-14 评估 biz-005/017/029、trap-005 四例实测）
    # - 模型有 tool_calls（含去重后裁剪）：合并守卫调用后追加单条消息
    # - 模型无 tool_calls 且守卫补充：追加单条合成 assistant（占位 rc）
    # - 空响应且无补充调用：保留正文（完整轨迹，闲聊/信息充分场景）
    if assistant_msg.get("tool_calls"):
        kept_ids = {c.get("id") for c in planned if c.get("id")}
        assistant_msg["tool_calls"] = [
            t for t in assistant_msg["tool_calls"] if t.get("id") in kept_ids
        ]
        if synth_calls:
            assistant_msg["tool_calls"] = [
                *(assistant_msg["tool_calls"]),
                *(_synth_tc(c) for c in synth_calls),
            ]
        if assistant_msg["tool_calls"]:
            fc_messages.append(assistant_msg)
    elif synth_calls:
        fc_messages.append({
            "role": "assistant",
            "content": "（系统补充的核验工具调用）",
            "reasoning_content": "（系统补充的核验工具调用）",
            "tool_calls": [_synth_tc(c) for c in synth_calls],
        })
    else:
        fc_messages.append(assistant_msg)

    logger.info("[planner] round=%d planned=%s should_continue=%s",
                rounds, planned, should_continue)
    return {
        "rounds": rounds,
        "planned_calls": planned,
        "should_continue": should_continue,
        "fc_messages": fc_messages,
        "skill_name": skill_name,
        "skill_instructions": skill_instructions,
        "skill_tool_names": skill_tool_names,
        "experiences": experiences,
        "history_context": history_context,
    }


def _dedupe_planned_calls(
    planned: list[dict[str, Any]],
    history_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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


def _plan_via_fc(
    fc_messages: list[dict[str, Any]],
    skill_tool_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """通过原生 FC 消息序列调用规划 LLM。返回 (planned, assistant_msg)。

    planned: [{"name", "arguments", "id"}]，id 为模型返回的 tool_call id
    （供 executor 的 tool 消息配对）。
    assistant_msg: 可直接回传续轮的原生 assistant 消息——content 与
    tool_calls 原样保留，reasoning_content 在模型返回时原样携带
    （DeepSeek 思考模式续轮强制要求回传；其他后端不返回则字段不携带，
    保持后端无关）。
    FC 通道为空时从正文抢救"文本形式"工具调用（评估失效模式 1）。
    """
    from agent.tools.schemas import build_openai_tools

    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["planner"])
    # Skill 工具子集过滤：空列表或 None = 全部工具。
    # list_skills / read_session_archive 是元工具（对标 MCP tools/list 与
    # 记忆还原通道），不受技能工具子集隔离限制，始终保留在 schema 中——
    # 否则 system prompt 广告了它们而 schema 里没有，
    # LLM 按提示调用会抛 "Unknown tool"，被反思模块误记为"工具失败教训"
    effective_tool_names = skill_tool_names or None
    if effective_tool_names:
        for meta_tool in ("list_skills", "read_session_archive"):
            if meta_tool not in effective_tool_names:
                effective_tool_names = [*effective_tool_names, meta_tool]
    tools_schema = build_openai_tools(tool_names=effective_tool_names)

    try:
        resp = client.chat.completions.create(
            model=settings["model"],
            messages=fc_messages,
            tools=tools_schema,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=1024,
        )
        record_llm_usage("planner", resp.usage)
    except (APITimeoutError, RateLimitError, APIConnectionError, APIError) as e:
        logger.exception("[planner] LLM Function Calling 调用失败 (%s)", type(e).__name__)
        raise _classify_llm_error(e) from e
    except Exception as e:
        logger.exception("[planner] LLM 未知异常")
        raise _classify_llm_error(e) from e

    msg = resp.choices[0].message
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or None}
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        assistant_msg["reasoning_content"] = reasoning
    if msg.tool_calls:
        assistant_msg["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name,
                          "arguments": tc.function.arguments or "{}"}}
            for tc in msg.tool_calls
        ]

    planned = []
    for tc in msg.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            planned.append({"name": tc.function.name, "arguments": args, "id": tc.id})
        except json.JSONDecodeError as e:
            logger.warning("[planner] failed to parse args for %s: %s", tc.function.name, e)
            continue
    if not planned:
        # 守卫：FC 通道为空时，从正文抢救"文本形式"的工具调用——部分模型会把
        # <tool_call>/[调用 xxx] 写进正文而非走 Function Calling（评估暴露的
        # 失效模式 1，详见 evals/FINDINGS.md）。抢救失败则维持"信息充分"判定。
        from agent.graph.planner_guard import rescue_text_tool_calls
        rescued = rescue_text_tool_calls(extract_content(msg) or "")
        if rescued:
            return rescued, assistant_msg
        logger.info("[planner] LLM decided no tool calls needed (info sufficient)")
    return planned, assistant_msg
