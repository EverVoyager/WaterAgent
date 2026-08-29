"""planner 守卫层：文本工具调用救援 + 声称核验闸 + 工具完成度检查。

三类失效模式来自首次系统级评估（evals/，62 条基线，2026-08-29）：
1. 模型在回复正文中"叙述"工具调用（<tool_call> / <｜DSML｜invoke> /
   [调用 xxx]）而非走 Function Calling 通道 → 第 1 轮零工具 → 误判闲聊
2. 用户口头声称预警等级时模型顺从"不用查了"，跳过数据核验（陷阱抵抗 33%）
3. 研判类漏调水情工具、预案类不调 generate_plan（工具召回 48.1%）

守卫是确定性管道逻辑，与提示词层修复（planner system prompt）互补：
书中方法论——表现问题先分三层假设（表层提示词 / 中层管道 / 深层模型），
管道层兜底对"换模型"最鲁棒。
"""
import json
import logging
import re
from typing import Any

from agent.tools.schemas import TOOL_PARAM_MODELS

logger = logging.getLogger(__name__)

_DATA_TOOLS = frozenset({"get_hydrology", "get_weather", "predict_runoff"})

# ====== 1. 文本工具调用救援 ======

# Hermes/DeepSeek 文本格式：<tool_call>{"name":..., "arguments":{...}}</tool_call>
_HERMES_RE = re.compile(r"<tool_call>\s*(\[?\{.*?\}\]?)\s*</tool_call>", re.DOTALL)
# XML 风格 invoke 块，含 DeepSeek DSML 变体（<｜DSML｜invoke name="x">）
_INVOKE_RE = re.compile(r"<(?:\S{0,12}?)?invoke\s+name=\"(\w+)\">(.*?)</(?:\S{0,12}?)?invoke>",
                        re.DOTALL)
_PARAM_RE = re.compile(
    r"<(?:\S{0,12}?)?parameter\s+name=\"(\w+)\"[^>]*>(.*?)</(?:\S{0,12}?)?parameter>",
    re.DOTALL,
)
# 中文叙述：[调用 xxx，参数：{...}] / [工具调用] xxx，参数：{...}
# 两种形态：括号包裹整体（[调用 xxx，…]）或仅包裹关键词（[工具调用] xxx，…）；
# 名字与 JSON 之间的中文填充（"关键词：…"等）用贪心填充符跳过
_NARRATE_RE = re.compile(
    r"[\[【]\s*(?:工具调用|调用)\s*[\]】]?\s*([a-z_]+)[^\{\}\n]*(\{[^{}\[\]]*\})?",
)

_LEVEL_TOKEN_RE = re.compile(r"[ⅠⅡⅢⅣ1234一二三四]\s*级")
# 等级声称线索：陈述既成事实、口语推测或催促跳过核验
# （"估计/据说"来自第 2 轮评估 trap-003 实测：'我估计有6000了，按Ⅱ级给出预案'）
_CLAIM_CUE_RE = re.compile(
    r"已达到|已达|已经达到|已发布|已经发布|肯定是|应该是|估计|据说|我猜|大概|听说"
    r"|直接按|按.{0,3}级|不用查|无需查|不用再查|别查",
)
_STATION_RE = re.compile(r"(吴堡|龙门|府谷)")

# 预案生成请求："生成/制定/编制/输出/安排 …… 预案/应急方案"
_PLAN_REQUEST_RE = re.compile(
    r"(生成|制定|编制|输出|安排)[^。？！?!\n]{0,15}"
    r"(应急预案|应急响应方案|处置预案|处置方案|应急方案|转移方案|预案)",
)
# 研判请求：需要实时水情 + 降雨两个数据源
_ASSESS_RE = re.compile(r"研判|防汛形势|洪水风险|防汛压力|风险评[估判]|综合评[估判]")

_LEVEL_CN_TO_CODE = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV",
                     "1": "I", "2": "II", "3": "III", "4": "IV",
                     "一": "I", "二": "II", "三": "III", "四": "IV"}


def _valid_call(name: Any, arguments: Any) -> dict[str, Any] | None:
    """校验救援出的调用：工具名必须真实存在，参数必须为 dict。"""
    if isinstance(name, str) and name in TOOL_PARAM_MODELS and isinstance(arguments, dict):
        return {"name": name, "arguments": arguments, "source": "text_rescue"}
    return None


def _dedupe_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple] = set()
    result = []
    for c in calls:
        try:
            sig = (c["name"], json.dumps(c["arguments"], sort_keys=True, ensure_ascii=False))
        except (TypeError, ValueError):
            sig = (c["name"], str(c["arguments"]))
        if sig not in seen:
            seen.add(sig)
            result.append(c)
    return result


def rescue_text_tool_calls(content: str) -> list[dict[str, Any]]:
    """从模型回复正文中抢救"文本形式"的工具调用，解析为结构化 planned_calls。

    覆盖三种观察到的叙述格式（评估报告 biz-007/008/010/011）：
    A. <tool_call>{json}</tool_call>（含 JSON 数组）
    B. XML/DSML invoke 参数块
    C. 中文叙述 [调用 xxx，参数：{...}]

    只有工具名真实存在且参数为 dict 的调用才被采纳；解析不出则返回空列表，
    调用方维持原判定（闲聊/信息充分），不影响正常路径。
    """
    if not content or not ("tool_call" in content or "invoke" in content or "调用" in content):
        return []

    calls: list[dict[str, Any]] = []

    # A. Hermes <tool_call>
    for m in _HERMES_RE.finditer(content):
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("tool")
            args = item.get("arguments")
            if args is None:
                args = item.get("parameters")
            call = _valid_call(name, args if args is not None else {})
            if call:
                calls.append(call)

    # B. XML/DSML invoke 块
    for m in _INVOKE_RE.finditer(content):
        name, body = m.group(1), m.group(2)
        args = {pm.group(1): pm.group(2).strip() for pm in _PARAM_RE.finditer(body)}
        call = _valid_call(name, args)
        if call:
            calls.append(call)

    # C. 中文叙述
    for m in _NARRATE_RE.finditer(content):
        name, args_json = m.group(1), m.group(2)
        args: dict = {}
        if args_json:
            try:
                parsed = json.loads(args_json)
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                pass
        call = _valid_call(name, args)
        if call:
            calls.append(call)

    calls = _dedupe_calls(calls)
    if calls:
        logger.warning(
            "[planner-guard] 文本工具调用救援：FC 通道为空，从正文解析出 %d 个调用: %s",
            len(calls), [c["name"] for c in calls],
        )
    return calls


# ====== 2. 声称核验闸（反讨好） ======

def user_claims_level(query: str) -> bool:
    """用户是否在 query 中以既成事实的口吻声称了预警等级/数据。

    对齐 τ²-bench trap tasks 设计：只有"声称 + 催促跳过核验"同时出现才触发，
    法规问答类（"启动Ⅱ级响应需要什么条件"）不受影响。
    """
    return bool(_LEVEL_TOKEN_RE.search(query) and _CLAIM_CUE_RE.search(query))


def _station_of(query: str) -> str | None:
    m = _STATION_RE.search(query)
    return m.group(1) if m else None


def enforce_claim_verification(query: str, planned: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """声称核验闸：query 声称等级但本轮未规划任何数据工具 → 强制追加核验调用。

    数据一旦进入 tool_results，synthesizer 已有的等级一致性门会把等级锚定到
    规则引擎真值——本闸只负责保证"数据在场"，定级交给既有机制。
    """
    if not user_claims_level(query):
        return planned
    planned_names = {c.get("name") for c in planned if isinstance(c, dict)}
    if planned_names & _DATA_TOOLS:
        return planned
    station = _station_of(query) or "吴堡"
    forced = [
        {"name": "get_hydrology", "arguments": {"station": station, "metric": "both"},
         "source": "claim_verification"},
        {"name": "get_weather", "arguments": {"location": f"{station}站", "hours": 24},
         "source": "claim_verification"},
    ]
    logger.warning(
        "[planner-guard] 声称核验闸触发：query 声称等级但无数据工具，强制追加核验调用",
    )
    return planned + forced


# ====== 3. 工具完成度检查 ======

def missing_required_tools(
    query: str,
    called_names: set[str],
    tool_results: dict[str, Any],
) -> list[dict[str, Any]]:
    """预案/研判类查询的关键工具完成度检查。

    在 planner 宣布"信息充分"（planned 为空）时调用；发现关键工具缺失则
    返回应强制补充的调用（由 planner 追加为本轮 planned，再走一轮）。
    - 预案请求：generate_plan 未调用过 → 用规则引擎定级补一次 generate_plan
      （等级取 compute_warning_level 真值，fallback 用 query 声称的等级）
    - 研判请求：实时水情（get_hydrology）与降雨（get_weather）缺一补一
    """
    calls: list[dict[str, Any]] = []

    if _PLAN_REQUEST_RE.search(query) and "generate_plan" not in called_names:
        level = ""
        if tool_results:
            level, _ = _compute_level(tool_results)
        if not level:
            m = _LEVEL_TOKEN_RE.search(query)
            level = _LEVEL_CN_TO_CODE.get(m.group(0)[0], "IV") if m else "IV"
        station = _station_of(query)
        calls.append({
            "name": "generate_plan",
            "arguments": {
                "warning_level": level,
                "affected_area": f"{station}河段" if station else "吕梁市",
                "population_at_risk": 0,
            },
            "source": "completeness",
        })

    if _ASSESS_RE.search(query):
        station = _station_of(query)
        if "get_hydrology" not in called_names:
            calls.append({
                "name": "get_hydrology",
                "arguments": {"station": station or "吴堡", "metric": "both"},
                "source": "completeness",
            })
        if "get_weather" not in called_names:
            calls.append({
                "name": "get_weather",
                "arguments": {"location": f"{station}站" if station else "吕梁市", "hours": 24},
                "source": "completeness",
            })

    if calls:
        logger.warning(
            "[planner-guard] 完成度闸触发：query 要求超出已调用工具，强制补充: %s",
            [c["name"] for c in calls],
        )
    return calls


def _compute_level(tool_results: dict[str, Any]) -> tuple[str, str]:
    """规则引擎真值（延迟导入避免模块加载顺序问题）。"""
    from agent.graph.synthesizer import compute_warning_level
    return compute_warning_level(tool_results)
