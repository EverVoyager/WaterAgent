"""Skill 包导入器。

支持两种格式：
1. .zip / .skill 压缩包（内含 SKILL.md）
2. 单个 .md 文件（SKILL.md 内容）

安全措施：
- ZIP 炸弹防护（解压后总大小限制、文件数限制）
- 路径穿越防护（../ 和绝对路径）
- 上传大小限制
- tool_names 白名单过滤

冲突策略（name 已存在时）：
- overwrite: 覆盖同名技能
- rename: 重命名（加 _imported_N 后缀）
- cancel: 报错（默认）
"""
import io
import logging
import zipfile
from dataclasses import dataclass

from agent.skills.matcher import invalidate_cache
from agent.skills.models import Skill, SkillCreate
from agent.skills.parser import (
    filter_tools_by_whitelist,
    parse_skill_md,
)
from agent.skills.store import create_skill, delete_skill, get_skill

logger = logging.getLogger(__name__)

# 安全限制
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB 上传大小
MAX_UNCOMPRESSED_SIZE = 50 * 1024 * 1024  # 50MB 解压后总大小
MAX_ZIP_ENTRIES = 1000  # ZIP 文件数上限
SKILL_MD_FILENAME = "SKILL.md"

# 支持的冲突策略
VALID_STRATEGIES = {"overwrite", "rename", "cancel"}


@dataclass
class ImportResult:
    """导入结果。"""

    skill: Skill | None
    action: str  # "created" / "overwritten" / "renamed"
    original_name: str
    final_name: str
    warnings: list[str]


def import_skill_from_md(
    content: str,
    conflict_strategy: str = "cancel",
) -> ImportResult:
    """从 SKILL.md 文本内容导入 Skill。

    Args:
        content: SKILL.md 文件内容
        conflict_strategy: 冲突策略 (overwrite/rename/cancel)

    Returns:
        ImportResult 导入结果

    Raises:
        ValueError: 解析失败、校验失败、冲突且策略为 cancel
    """
    if conflict_strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"无效的冲突策略: {conflict_strategy}，可选: {VALID_STRATEGIES}"
        )

    parsed = parse_skill_md(content)

    # 兼容 Claude Skills name 格式：连字符转下划线
    normalized_name = parsed.name.replace("-", "_")

    # 白名单过滤工具
    filtered_tools, tool_warnings = filter_tools_by_whitelist(
        parsed.raw_tool_names, Skill.BUILTIN_TOOLS
    )
    warnings = list(tool_warnings)

    # 冲突检测
    original_name = normalized_name
    final_name = normalized_name
    action = "created"

    existing = get_skill(normalized_name)
    if existing is not None:
        if conflict_strategy == "overwrite":
            delete_skill(normalized_name)
            action = "overwritten"
        elif conflict_strategy == "rename":
            final_name = _generate_unique_name(normalized_name)
            action = "renamed"
        else:  # cancel
            raise ValueError(
                f"Skill '{normalized_name}' 已存在，请选择覆盖、重命名或取消"
            )

    # 构建 SkillCreate（触发模型校验）
    try:
        req = SkillCreate(
            name=final_name,
            description=parsed.description,
            instructions=parsed.instructions,
            tool_names=filtered_tools,
            enabled=True,
        )
    except Exception as e:
        raise ValueError(f"Skill 数据校验失败: {e}") from e

    skill = create_skill(req)
    invalidate_cache()

    logger.info(
        "[skill-import] 导入成功: %s -> %s (action=%s, tools=%s)",
        original_name, final_name, action, filtered_tools or "all",
    )

    return ImportResult(
        skill=skill,
        action=action,
        original_name=original_name,
        final_name=final_name,
        warnings=warnings,
    )


def import_skill_from_zip(
    zip_bytes: bytes,
    conflict_strategy: str = "cancel",
) -> ImportResult:
    """从 ZIP 压缩包导入 Skill。

    ZIP 内必须包含 SKILL.md 文件（根目录或子目录均可）。

    Args:
        zip_bytes: ZIP 文件二进制内容
        conflict_strategy: 冲突策略

    Returns:
        ImportResult 导入结果

    Raises:
        ValueError: ZIP 损坏、无 SKILL.md、安全校验失败
    """
    if len(zip_bytes) > MAX_UPLOAD_SIZE:
        raise ValueError(
            f"文件过大: {len(zip_bytes)} bytes，超过限制 {MAX_UPLOAD_SIZE} bytes"
        )

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as e:
        raise ValueError(f"无效的 ZIP 文件: {e}") from e

    # 安全校验
    _validate_zip_safety(zf)

    # 查找 SKILL.md
    skill_md_path = _find_skill_md(zf)
    if skill_md_path is None:
        raise ValueError("ZIP 包内未找到 SKILL.md 文件")

    try:
        content = zf.read(skill_md_path).decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"SKILL.md 编码错误（需 UTF-8）: {e}") from e

    return import_skill_from_md(content, conflict_strategy=conflict_strategy)


def _find_skill_md(zf: zipfile.ZipFile) -> str | None:
    """在 ZIP 中查找 SKILL.md 文件（不区分大小写）。

    优先返回路径最短的（根目录优先）。
    """
    target = SKILL_MD_FILENAME.upper()
    candidates = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue  # 跳过目录
        basename = name.rsplit("/", 1)[-1]
        if basename.upper() == target:
            candidates.append(name)

    if not candidates:
        return None

    candidates.sort(key=len)
    return candidates[0]


def _validate_zip_safety(zf: zipfile.ZipFile) -> None:
    """ZIP 安全校验：炸弹防护 + 路径穿越。"""
    total_uncompressed = 0

    for entries, info in enumerate(zf.infolist(), start=1):
        if entries > MAX_ZIP_ENTRIES:
            raise ValueError(f"ZIP 文件数超过限制: {MAX_ZIP_ENTRIES}")

        # 路径穿越防护
        if ".." in info.filename or info.filename.startswith("/"):
            raise ValueError(f"ZIP 含非法路径: {info.filename}")

        total_uncompressed += info.file_size
        if total_uncompressed > MAX_UNCOMPRESSED_SIZE:
            raise ValueError(
                f"ZIP 解压后总大小超过限制: {MAX_UNCOMPRESSED_SIZE} bytes"
            )


def _generate_unique_name(original: str) -> str:
    """生成唯一的技能名（加 _imported_N 后缀）。"""
    base = original
    n = 1
    while get_skill(f"{base}_imported_{n}") is not None:
        n += 1
    return f"{base}_imported_{n}"
