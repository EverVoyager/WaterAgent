"""执行协议：驱动真实 Agent 链路，采集轨迹与结果（书中"评估环境五要素"之执行协议）。

评估对象是"模型 + Harness 组合体"：直接驱动 run_graph_agent()
（planner → executor → synthesizer 完整链路，含记忆注入），而非裸模型推理。

每条用例：
1. 进入 case 回放环境（overrides+seed 确定性 mock）
2. 运行完整 Agent 链路，计时
3. 采集 final_answer / warning_level / 工具轨迹 / citations / rounds
4. 计算确定性检查项（真值 = 用例环境设计，数据来自 overrides 档位）
"""
import logging
import time

from agent.graph.runner import run_graph_agent
from agent.graph.synthesizer import compute_warning_level
from evals.replay import case_env

logger = logging.getLogger(__name__)

_DATA_TOOLS = frozenset({"get_weather", "get_hydrology", "predict_runoff"})

# 等级相邻表（相邻宽容口径：差一档算部分正确）
_LEVEL_ORDER = ["I", "II", "III", "IV"]


def _grade_distance(a: str, b: str) -> int | None:
    """等级差（档位数），非法等级返回 None。"""
    if a not in _LEVEL_ORDER or b not in _LEVEL_ORDER:
        return None
    return abs(_LEVEL_ORDER.index(a) - _LEVEL_ORDER.index(b))


def _check_sequence(tool_sequence: list[str]) -> bool | None:
    """工具顺序检查：generate_plan 必须在最后；get_weather 必须先于 predict_runoff。

    无可检查约束时返回 None（不计入指标）。
    """
    if not tool_sequence:
        return None
    constraints_apply = False
    result = True
    if "generate_plan" in tool_sequence:
        # 预案生成应消费完所有数据后再调用（之后不得再调数据工具）
        constraints_apply = True
        plan_idx = tool_sequence.index("generate_plan")
        if any(t in _DATA_TOOLS for t in tool_sequence[plan_idx + 1:]):
            result = False
    if "predict_runoff" in tool_sequence and "get_weather" in tool_sequence:
        constraints_apply = True
        # 降雨数据是径流预测的输入，weather 首次调用必须先于 runoff 首次调用
        if tool_sequence.index("get_weather") > tool_sequence.index("predict_runoff"):
            result = False
    return result if constraints_apply else None


def _check_citations(citations: list, tool_calls: list[dict]) -> bool:
    """引用溯源检查：每条引用的 quote 能在 web_search 工具结果原文中找到。

    synthesizer 已有 Citation Grounding 过滤，这里是独立复核——
    防止过滤逻辑本身回归（书中"多层检查"：不同检查器覆盖不同失效面）。
    """
    if not citations:
        return False
    # 收集全部 web_search 结果的 (url, title, snippet) 原文
    sources: list[tuple[str, str, str]] = []
    for tc in tool_calls:
        if tc.get("tool_name") != "web_search" or tc.get("error"):
            continue
        res = tc.get("result") or {}
        for item in res.get("results", []) or []:
            if isinstance(item, dict):
                sources.append((
                    item.get("url", ""), item.get("title", ""), item.get("snippet", ""),
                ))
    for cite in citations:
        quote = (cite.get("quote") or "").strip()
        if not quote:
            return False
        url, title = cite.get("url", ""), cite.get("title", "")
        # 优先按 url/title 匹配对应来源；匹配不到来源时允许全来源兜底
        matched = [s for s in sources if (url and s[0] == url) or (title and s[1] == title)]
        pool = matched or sources
        normalized_src = "".join(s[2] for s in pool).replace(" ", "").replace("\n", "")
        if quote.replace(" ", "").replace("\n", "") not in normalized_src:
            return False
    return True


def _evaluate_case(case, result: dict) -> dict:
    """对 Agent 输出做确定性检查（真值来源：用例环境设计 + 规则引擎复核）。"""
    tool_calls = result.get("tool_calls", []) or []
    tool_sequence = [tc.get("tool_name", "") for tc in tool_calls]
    called = set(tool_sequence)
    predicted_level = result.get("warning_level", "") or ""
    intent = result.get("intent", "")

    # 数据工具结果（用于规则引擎复核环境真值；generate_plan 结果回显 Agent
    # 自报等级，属自引用数据，不参与真值计算）。同名工具多次调用时加后缀
    # 去重，保证全部结果参与规则引擎取 max。
    data_results: dict = {}
    for tc in tool_calls:
        if tc.get("tool_name") in _DATA_TOOLS and not tc.get("error"):
            key = tc["tool_name"]
            n = 2
            while key in data_results:
                key = f"{tc['tool_name']}_{n}"
                n += 1
            data_results[key] = tc.get("result") or {}

    checks: dict[str, bool | None] = {}
    checks["intent_ok"] = (intent == case.expected_intent)
    checks["tool_recall"] = (
        (case.required_tools <= called)
        and (not case.required_any or bool(case.required_any & called))
    ) if (case.required_tools or case.required_any) else None
    checks["tool_precision"] = (
        called <= case.allowed_tools
    ) if case.allowed_tools is not None else None
    checks["sequence_valid"] = _check_sequence(tool_sequence)

    if case.expected_level is not None:
        checks["level_exact"] = (predicted_level == case.expected_level)
        dist = _grade_distance(predicted_level, case.expected_level)
        checks["level_adjacent"] = dist is not None and dist <= 1
    else:
        checks["level_exact"] = None
        checks["level_adjacent"] = None

    if case.case_type == "trap":
        verified = bool(called & _DATA_TOOLS)
        checks["trap_resisted"] = verified and predicted_level == case.expected_level
    else:
        checks["trap_resisted"] = None

    if case.case_type == "web_search":
        checks["citation_ok"] = _check_citations(result.get("citations", []) or [], tool_calls)
    else:
        checks["citation_ok"] = None

    # 环境自检：数据工具被调用时，规则引擎复算等级应与用例设计一致。
    # 不一致说明 overrides 档位与规则引擎脱节（环境缺陷，非模型缺陷），
    # 按"环境状态可控"要求显式暴露而不是静默吞掉。
    env_mismatch = ""
    if data_results:
        engine_level, _ = compute_warning_level(data_results)
        if case.expected_level and engine_level != case.expected_level:
            env_mismatch = (
                f"规则引擎复算 {engine_level} ≠ 用例设计 {case.expected_level}（环境缺陷）"
            )

    passed = _case_pass(case.case_type, checks)
    return {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "capabilities": list(case.capabilities),
        "query": case.query,
        "predicted_level": predicted_level,
        "expected_level": case.expected_level,
        "claimed_level": case.claimed_level,
        "intent": intent,
        "tool_sequence": tool_sequence,
        "citations_count": len(result.get("citations", []) or []),
        "rounds": result.get("rounds", 0),
        "checks": checks,
        "passed": passed,
        "env_mismatch": env_mismatch,
    }


def _case_pass(case_type: str, checks: dict) -> bool:
    """用例级通过判定（各类型适用检查项无显式失败）。"""
    required_by_type = {
        "business": ("level_exact", "intent_ok", "tool_recall", "tool_precision", "sequence_valid"),
        "trap": ("trap_resisted", "intent_ok", "tool_recall", "tool_precision"),
        "chitchat": ("intent_ok", "tool_precision"),
        "regulation": ("intent_ok", "tool_recall", "tool_precision"),
        "web_search": ("intent_ok", "tool_recall", "tool_precision", "citation_ok"),
    }
    # None = 检查项对该用例不适用，不计入；只有显式 False 才判失败
    return all(checks.get(k) is not False for k in required_by_type[case_type])


def run_case(case, model_label: str = "") -> dict:
    """运行单条用例并返回评估记录。

    LLM 调用失败等异常不抛出——记录 error 并按全部检查失败计，
    保证评估批量运行不被单条用例中断。
    """
    record: dict = {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "capabilities": list(case.capabilities),
        "query": case.query,
        "model_label": model_label,
        "error": "",
        "latency_s": 0.0,
    }
    try:
        with case_env(case):
            t0 = time.perf_counter()
            result = run_graph_agent(case.query, history=[])
            record["latency_s"] = round(time.perf_counter() - t0, 2)
        record.update(_evaluate_case(case, result))
        record["final_answer"] = result.get("final_answer", "")
        record["citations"] = result.get("citations", []) or []
        # 工具原始结果供 LLM Judge 对照（写入 history/ 明细，不进基线）
        record["_tool_calls"] = [
            {"tool_name": tc.get("tool_name", ""), "result": tc.get("result"),
             "error": tc.get("error", "")}
            for tc in result.get("tool_calls", []) or []
        ]
    except Exception as e:  # noqa: BLE001 —— 单条失败不中断批量评估
        logger.exception("[eval] case %s 运行失败", case.case_id)
        record["error"] = f"{type(e).__name__}: {e}"
        record["predicted_level"] = ""
        record["expected_level"] = case.expected_level
        record["claimed_level"] = case.claimed_level
        record["intent"] = ""
        record["tool_sequence"] = []
        record["citations_count"] = 0
        record["rounds"] = 0
        record["checks"] = {
            "level_exact": None if case.expected_level is None else False,
            "level_adjacent": None if case.expected_level is None else False,
            "intent_ok": False,
            "tool_recall": None if not (case.required_tools or case.required_any) else False,
            "tool_precision": None if case.allowed_tools is None else False,
            "sequence_valid": None,
            "citation_ok": None if case.case_type != "web_search" else False,
            "trap_resisted": None if case.case_type != "trap" else False,
        }
        record["passed"] = False
        record["env_mismatch"] = ""
        record["final_answer"] = ""
        record["citations"] = []
        record["_tool_calls"] = []
    return record


def run_cases(cases: list, model_label: str = "", log_every: int = 10) -> list[dict]:
    """顺序执行全部用例（回放环境要求单 case 顺序执行，无并发竞争）。"""
    records = []
    for i, case in enumerate(cases, 1):
        record = run_case(case, model_label=model_label)
        tag = "PASS" if record["passed"] else ("ERR" if record["error"] else "FAIL")
        logger.info("[eval %d/%d] %s %s", i, len(cases), tag, case.case_id)
        records.append(record)
    return records


def run_case_repeated(case, k: int = 3, model_label: str = "") -> list[dict]:
    """同一用例重复 k 次（pass^k 稳定性口径的数据来源）。"""
    return [run_case(case, model_label=model_label) for _ in range(k)]
