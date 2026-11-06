"""记忆安全闸：提示词注入、敏感信息和领域事实校验。"""
import logging
import re

from agent.utils import WARNING_THRESHOLDS

logger = logging.getLogger(__name__)

# ====== 写入安全闸 ======

# 提示词注入攻击载荷特征（写入侧安全闸门，借鉴 Hermes 的 prompt 注入扫描）
# 恶意用户可通过"请记住：忽略所有指令"把注入攻击持久化到记忆
_UNSAFE_MEMORY_PATTERNS = (
    "忽略所有", "忽略以上", "忽略之前", "忽略上述", "无视所有", "无视指令",
    "ignore all", "ignore previous", "ignore above",
    "系统提示", "系统指令", "system prompt", "开发者模式", "developer mode",
    "从现在起你", "你是一个新的", "新人设", "最高优先级", "无条件服从",
    "越狱", "jailbreak",
)


def _is_unsafe_memory_content(content: str) -> bool:
    """检测记忆内容是否含提示词注入攻击载荷（所有记忆类型通用）。"""
    if not content:
        return False
    lowered = content.lower()
    return any(pattern in lowered for pattern in _UNSAFE_MEMORY_PATTERNS)

# 敏感信息特征（对齐 Codex memories 的 redaction：凭据/个人隐私不写入记忆）
_SENSITIVE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),                    # API key
    re.compile(r'''(?:api[_-]?key|apikey)["']?\s*[:=]\s*["']?\S{8,}''', re.I),
    re.compile(r'''(?:password|passwd|pwd)["']?\s*[:=]\s*["']?\S{6,}''', re.I),
    re.compile(r'''(?:secret|token)["']?\s*[:=]\s*["']?\S{8,}''', re.I),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{16,}"),
    re.compile(r"1[3-9]\d{9}"),                            # 手机号
    re.compile(r"\d{17}[\dXx]"),                           # 身份证号
)


def _is_sensitive_content(content: str) -> bool:
    """检测记忆内容是否含敏感信息（API key/密码/token/手机号/身份证）。"""
    if not content:
        return False
    return any(p.search(content) for p in _SENSITIVE_PATTERNS)

# ====== 第四道写入安全闸：领域事实校验（与规则引擎单一同源）======
# 反思 LLM 是会出错的写入者：一条错误阈值断言（如"流量超3000发Ⅰ级"——实际Ⅰ级需≥5000）
# 一旦入库，会被后续检索系统性注入放大（越用越错）。写入前对照 WARNING_THRESHOLDS
# 校验"数值 ↔ 等级"映射。校验失败的记忆拒写并留 warning 审计痕迹。

# 等级词元：Unicode 罗马数字 / ASCII 罗马数字 / 中文数字（"第X级/一级别"排除）
_LEVEL_RE = re.compile(r"(?<!第)(Ⅰ|Ⅱ|Ⅲ|Ⅳ|IV|III|II|I|一|二|三|四)\s*级(?!别)")

_TOKEN_TO_LEVEL = {
    "Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV",
    "一": "I", "二": "II", "三": "III", "四": "IV",
    "I": "I", "II": "II", "III": "III", "IV": "IV",
}

# 子句切分：句号/分号/逗号等（子句级校验，避免跨句误配）
_CLAUSE_SPLIT_RE = re.compile(r"[。！？；\n，,、]")

# 信号关键词：流量 / 降雨 / 水位（水位涉及 OR 多判据，单独处理）
_FLOW_HINT_RE = re.compile(r"流量|m[³3]/s|m3/s|立方米每秒")

_RAIN_HINT_RE = re.compile(r"降雨|雨量|毫米|mm", re.I)

_WATER_HINT_RE = re.compile(r"水位|警戒|保证")

# 数字提取：优先取带单位的（"150毫米"），避免"24小时降雨…150毫米"误取 24
_NUM_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:毫米|mm|m[³3]/s|m3/s|立方米每秒)", re.I)

_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _flow_level_for(value: float) -> str:
    """流量值 → 等级（与 compute_warning_level 同阈值单一同源）。"""
    if value >= WARNING_THRESHOLDS["flow_level1"]:
        return "I"
    if value >= WARNING_THRESHOLDS["flow_level2"]:
        return "II"
    if value >= WARNING_THRESHOLDS["flow_level3"]:
        return "III"
    return "IV"


def _rain_level_for(value: float) -> str | None:
    """降雨值 → 等级。规则引擎只有 Ⅰ/Ⅱ 级降雨判据，无判据返回 None（跳过校验）。"""
    if value > WARNING_THRESHOLDS["rain_level1"]:
        return "I"
    if WARNING_THRESHOLDS["rain_level2"] <= value <= WARNING_THRESHOLDS["rain_level1"]:
        return "II"
    return None


def _is_factual_error(content: str) -> str | None:
    """校验记忆中的预警等级阈值断言，返回冲突子句（用于日志），无冲突返回 None。

    保守策略（宁漏勿错，防误杀合法记忆）：
    - 子句同时出现水位/警戒/保证关键词 → 跳过（OR 多判据下单信号不足以裁定等级）
    - 子句同时出现流量和降雨信号 → 跳过（同上）
    - 降雨值低于Ⅱ级判据 → 跳过（规则引擎无 Ⅲ/Ⅳ 级降雨标准）
    - 数字明显不在量纲范围（<5 或 >100000，如年份）→ 跳过
    """
    if not content:
        return None
    for clause in _CLAUSE_SPLIT_RE.split(content):
        level_m = _LEVEL_RE.search(clause)
        if not level_m:
            continue
        claimed = _TOKEN_TO_LEVEL[level_m.group(1)]
        if _WATER_HINT_RE.search(clause):
            continue
        has_flow, has_rain = bool(_FLOW_HINT_RE.search(clause)), bool(_RAIN_HINT_RE.search(clause))
        if has_flow and has_rain:
            continue
        num_m = _NUM_UNIT_RE.search(clause) or _NUM_RE.search(clause)
        if not num_m:
            continue
        value = float(num_m.group(1))
        if has_flow:
            if not (5 <= value <= 100000):
                continue
            if _flow_level_for(value) != claimed:
                return clause.strip()
        elif has_rain:
            expected = _rain_level_for(value)
            if expected is not None and expected != claimed:
                return clause.strip()
    return None


def _gate_violation(*texts: str) -> str | None:
    """对一段待写记忆执行全部文本安全闸，返回违规描述（事实闸返回冲突子句），通过返回 None。"""
    for text in texts:
        if not text:
            continue
        if _is_unsafe_memory_content(text):
            return f"注入载荷: {text[:60]}"
        if _is_sensitive_content(text):
            return f"敏感信息: {text[:60]}"
        factual = _is_factual_error(text)
        if factual:
            return f"事实冲突: {factual[:60]}"
    return None


def check_memory_safety(*texts: str) -> str | None:
    """公开安全闸入口：对任意来源（反思/治理 API/Curator）的待写记忆内容
    执行三道文本闸（提示词注入/敏感信息/领域事实），通过返回 None。

    手动治理路径必须与自动反思路径过同样的闸，避免"治理面比写入面更宽松"。
    """
    return _gate_violation(*texts)
