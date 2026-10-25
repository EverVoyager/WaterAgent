"""工具执行节点：并发调度、数据注入与结果校验。"""
import contextvars
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from agent.graph.cache import _cached_execute_tool
from agent.graph.state import AgentState

logger = logging.getLogger(__name__)

# 工具执行线程池（P3 并发执行）
_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool-exec")


def _execute_one_tool(
    call: dict[str, Any],
    idx: int,
    weather_rainfall_mm: float | None,
    weather_series: list[dict] | None,
    round_num: int,
    existing_keys: set,
    duplicate_names: set | None = None,
) -> dict[str, Any]:
    """执行单个工具调用，返回结构化记录。

    封装为独立函数以支持 P3 并发执行。包含 P6 缓存命中、跨工具数据流注入。
    """
    name = call.get("name", "")
    args = dict(call.get("arguments", {}) or {})
    error = ""
    result: dict[str, Any] = {}

    if not name:
        return {"name": name, "idx": idx, "args": args, "result": result,
                "error": "empty tool name", "result_key": f"empty_{idx}",
                "is_weather": False}

    # 跨工具数据流：predict_runoff 自动注入降雨量
    if name == "predict_runoff" and weather_rainfall_mm is not None:
        if args.get("rainfall_mm") is None:
            args["rainfall_mm"] = weather_rainfall_mm
        if weather_series and args.get("rainfall_series") is None:
            args["rainfall_series"] = weather_series
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

    # 返回值出口闸：schema 违规按工具失败处理（fail-fast），不让坏数据
    # 进入 tool_results——下游规则引擎/跨工具注入按 key 直读，字段漂移
    # 与其 KeyError 或静默算错，不如在边界拦截后交给 planner 重规划。
    if not error:
        from agent.tools.schemas import validate_tool_result
        violation = validate_tool_result(name, result)
        if violation:
            logger.warning("[executor] 工具返回值校验失败 %s: %s", name, violation)
            error = f"result_schema: {violation}"
            result = {}

    # 累积结果 key：跨轮同名（existing_keys）或本轮重复规划（duplicate_names）
    # 时加 idx 后缀，保证同轮多个同名工具调用结果不互相覆盖（对齐 OpenAI
    # tool_call_id 唯一配对语义）
    needs_suffix = name in existing_keys or name in (duplicate_names or set())
    result_key = f"{name}_{idx}" if needs_suffix else name

    return {
        "name": name,
        "idx": idx,
        "args": args,
        "result": result,
        "error": error,
        "result_key": result_key,
        "tc_id": call.get("id", ""),
        "is_weather": name == "get_weather" and isinstance(result, dict),
    }


def _collect_weather_context(tool_results: dict[str, Any]) -> tuple[float | None, list[dict] | None]:
    """从历史工具结果中提取 get_weather 的降雨数据（用于跨工具数据流注入）。"""
    for k, v in reversed(list(tool_results.items())):
        if k.startswith("get_weather") and isinstance(v, dict) and v.get("total_rainfall_mm") is not None:
            rainfall_mm = float(v["total_rainfall_mm"])
            series = v.get("series")
            return rainfall_mm, series
    return None, None


def _partition_stages(planned: list[dict[str, Any]]) -> tuple[list[tuple], list[tuple]]:
    """将计划工具调用分为两阶段：stage1 跑 get_weather，stage2 跑其余。

    仅当本轮同时包含 get_weather 和 predict_runoff 时才分阶段（保证降雨数据注入）。
    否则全部工具放入 stage2 并发执行。
    """
    valid = [(i, c) for i, c in enumerate(planned) if isinstance(c, dict)]
    has_weather = any(c.get("name") == "get_weather" for _, c in valid)
    has_runoff = any(c.get("name") == "predict_runoff" for _, c in valid)

    if has_weather and has_runoff:
        stage1 = [(i, c) for i, c in valid if c.get("name") == "get_weather"]
        stage2 = [(i, c) for i, c in valid if c.get("name") != "get_weather"]
    else:
        stage1 = []
        stage2 = valid
    return stage1, stage2


def _run_stage(
    stage_items: list[tuple],
    weather_rainfall_mm: float | None,
    weather_series: list[dict] | None,
    round_num: int,
    existing_keys: set,
    duplicate_names: set,
) -> list[dict[str, Any]]:
    """执行单个阶段的工具调用（单工具串行，多工具并发）。

    结果按 planned 中的原始顺序（idx）返回，与并发完成顺序无关
    （对齐 OpenAI tool_calls 协议：结果顺序与调用列表一致）。
    """
    if not stage_items:
        return []
    # 单工具直接串行，避免线程池开销
    if len(stage_items) == 1:
        i, c = stage_items[0]
        return [_execute_one_tool(c, i, weather_rainfall_mm, weather_series,
                                  round_num, existing_keys, duplicate_names)]
    # 多工具并发
    # 每个任务独立 copy_context：contextvars（含 request_id 日志上下文）不随
    # 线程池自动继承，拷贝后线程内工具日志与调用方同上下文；注意不能多个
    # 任务共享同一个 Context（并发 run 会抛 RuntimeError），必须逐任务拷贝
    futures = [
        _TOOL_EXECUTOR.submit(
            contextvars.copy_context().run,
            _execute_one_tool, c, i, weather_rainfall_mm, weather_series,
            round_num, existing_keys, duplicate_names,
        )
        for i, c in stage_items
    ]
    results = [fut.result() for fut in as_completed(futures)]
    # 按原始调用顺序（idx）排序，与线程完成顺序无关
    results.sort(key=lambda r: r["idx"])
    return results


def _apply_stage_results(
    results: list[dict[str, Any]],
    tool_results: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    existing_keys: set,
    round_num: int,
    update_weather: bool = False,
) -> tuple[float | None, list[dict] | None]:
    """把阶段执行结果回填到 tool_results 和 tool_calls。

    Args:
        update_weather: 若 True，从 get_weather 结果中更新降雨数据（供下一阶段注入）。
    Returns:
        (weather_rainfall_mm, weather_series) — 仅在 update_weather=True 时有意义。
    """
    weather_rainfall_mm = None
    weather_series: list[dict] | None = None
    for r in results:
        tool_results[r["result_key"]] = r["result"]
        tool_calls.append({
            "tool_name": r["name"],
            "arguments": r["args"],
            "result": r["result"],
            "error": r["error"],
            "round": round_num,
            "tc_id": r.get("tc_id", ""),
        })
        if update_weather and r["is_weather"]:
            if r["result"].get("total_rainfall_mm") is not None:
                weather_rainfall_mm = float(r["result"]["total_rainfall_mm"])
            if r["result"].get("series") is not None:
                weather_series = r["result"]["series"]
        existing_keys.add(r["result_key"])
    return weather_rainfall_mm, weather_series


def executor_node(state: AgentState) -> dict[str, Any]:
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
    prev_len = len(tool_calls)

    if not planned:
        return {"tool_results": tool_results, "tool_calls": tool_calls}

    # 本轮内重复规划的同名工具：result_key 需加 idx 后缀避免互相覆盖
    name_counts = Counter(c.get("name", "") for c in planned if isinstance(c, dict))
    duplicate_names = {n for n, cnt in name_counts.items() if cnt > 1}

    # 查找历史 get_weather 结果（用于跨工具数据流）
    weather_rainfall_mm, weather_series = _collect_weather_context(tool_results)

    # 分阶段执行
    stage1, stage2 = _partition_stages(planned)

    # 阶段 1：执行 weather（如果有的话）
    stage1_results = _run_stage(
        stage1, weather_rainfall_mm, weather_series, round_num, existing_keys,
        duplicate_names,
    )
    # 同轮 get_weather 结果立即更新，供 predict_runoff 使用
    new_rain, new_series = _apply_stage_results(
        stage1_results, tool_results, tool_calls, existing_keys,
        round_num, update_weather=True,
    )
    if new_rain is not None:
        weather_rainfall_mm = new_rain
    if new_series is not None:
        weather_series = new_series

    # 阶段 2：执行其余工具（含注入后的 predict_runoff）
    stage2_results = _run_stage(
        stage2, weather_rainfall_mm, weather_series, round_num, existing_keys,
        duplicate_names,
    )
    _apply_stage_results(
        stage2_results, tool_results, tool_calls, existing_keys, round_num,
    )

    # 工具结果转原生 tool 消息追加（与 assistant.tool_calls 按 id 配对）。
    # 错误信息附可操作建议（对齐 Anthropic 工具设计指南），超长结果截断
    tool_msgs = []
    for tc in tool_calls[prev_len:]:
        if tc.get("error"):
            content = json.dumps(
                {"error": tc["error"],
                 "hint": "可调整参数重试、改查其他站点，或基于已有信息研判"},
                ensure_ascii=False,
            )
        else:
            content = json.dumps(tc.get("result", {}), ensure_ascii=False)
        if len(content) > 4000:
            content = content[:4000] + '...(截断，如需更多细节请缩小查询范围)'
        tool_msgs.append({
            "role": "tool",
            "tool_call_id": tc.get("tc_id") or f"call_orphan_{round_num}_{len(tool_msgs)}",
            "content": content,
        })

    return {
        "tool_results": tool_results,
        "tool_calls": tool_calls,
        "fc_messages": list(state.get("fc_messages", [])) + tool_msgs,
    }
