"""SKILL.md 解析器（兼容 Claude Skills 开放标准）。

解析 SKILL.md 文件结构：
- YAML frontmatter（--- 包裹）：name + description 必填，allowed-tools/license 可选
- Markdown 正文：行为指令（加载后注入 prompt）

参考规范：
- https://www.mdskills.ai/zh/specs/skill-md
- Claude Code skills documentation
"""
import logging
import re
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)

# frontmatter 正则：匹配 --- 包裹的 YAML 块 + 后续正文
_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n?(.*)$",
    re.DOTALL,
)


@dataclass
class ParsedSkill:
    """SKILL.md 解析结果。"""

    name: str
    description: str
    instructions: str
    raw_tool_names: list[str] = field(default_factory=list)  # 原始 allowed-tools（未过滤）
    license: str | None = None
    warnings: list[str] = field(default_factory=list)  # 解析告警


def parse_skill_md(content: str) -> ParsedSkill:
    """解析 SKILL.md 内容。

    Args:
        content: SKILL.md 文件的文本内容

    Returns:
        ParsedSkill 解析结果

    Raises:
        ValueError: 缺少 frontmatter、缺少必填字段、或 YAML 解析失败
    """
    if not content or not content.strip():
        raise ValueError("SKILL.md 内容为空")

    match = _FRONTMATTER_RE.match(content)
    if not match:
        raise ValueError("SKILL.md 缺少 YAML frontmatter（应以 --- 开头）")

    yaml_text, body = match.groups()

    try:
        meta = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"frontmatter YAML 解析失败: {e}") from e

    if not isinstance(meta, dict):
        raise ValueError("frontmatter 必须是 YAML 键值对")

    # 必填字段
    name = meta.get("name")
    description = meta.get("description")
    if not name or not isinstance(name, str):
        raise ValueError("frontmatter 缺少必填字段: name")
    if not description or not isinstance(description, str):
        raise ValueError("frontmatter 缺少必填字段: description")

    # 可选字段
    license_ = meta.get("license")
    raw_tools = _parse_allowed_tools(meta.get("allowed-tools"))

    # 正文作为 instructions
    instructions = body.strip()
    if not instructions:
        # frontmatter 后无正文，用 description 兜底
        instructions = description.strip()

    return ParsedSkill(
        name=name.strip(),
        description=description.strip(),
        instructions=instructions,
        raw_tool_names=raw_tools,
        license=license_ if isinstance(license_, str) else None,
    )


def _parse_allowed_tools(value) -> list[str]:
    """解析 allowed-tools 字段。

    支持两种格式：
    - 字符串（空格分隔）: "Bash(python:*) Read Write"
    - 列表: ["Bash(python:*)", "Read", "Write"]

    返回原始工具名列表（未做白名单过滤）。
    """
    if not value:
        return []

    if isinstance(value, str):
        return value.split()

    if isinstance(value, list):
        return [str(v) for v in value if v]

    return []


def extract_tool_name(raw: str) -> str:
    """从 Claude 格式的工具名中提取纯工具名（小写）。

    例如:
        "Bash(python:*)" -> "bash"
        "Read" -> "read"
        "get_hydrology" -> "get_hydrology"
    """
    name = raw.split("(")[0].strip()
    return name.lower()


def filter_tools_by_whitelist(
    raw_tools: list[str],
    builtin_tools: set,
) -> tuple:
    """白名单过滤工具名。

    Args:
        raw_tools: 原始工具名列表（可能含 Claude 格式如 Bash(python:*)）
        builtin_tools: 项目内置工具白名单

    Returns:
        (filtered_tools, warnings): 过滤后的工具名列表 + 被丢弃项的告警
    """
    filtered = []
    warnings = []
    for raw in raw_tools:
        name = extract_tool_name(raw)
        if name in builtin_tools:
            filtered.append(name)
        else:
            warnings.append(f"工具 '{raw}' 不在内置白名单中，已忽略")
    return filtered, warnings
