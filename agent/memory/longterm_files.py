"""长期记忆文件：安全路径、原子读写、主题元数据和索引维护。"""
import logging
import re
import threading
from datetime import datetime
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# append 查重阈值：新内容的二元组已有内容覆盖比例超过此值视为重复，跳过
_APPEND_DUP_THRESHOLD = 0.85

# update 整体替换门槛：既有内容的二元组被新内容保留比例超过此值才允许替换，
# 否则降级为 append（防 LLM 误用 update 时一句话抹掉累积多轮的记忆）
_UPDATE_REPLACE_THRESHOLD = 0.6

# 索引文件中每个主题的摘要行长度
_TOPIC_SUMMARY_LEN = 60

# 主题文件 frontmatter（created/updated 日期，供时效判断与治理展示）
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)[ \t]*\n---[ \t]*\n?", re.S)

_DATE_VALUE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_lock = threading.RLock()


def _memory_dir() -> Path:
    """Agent 自动记忆目录（项目根 memory/，可经 MEMORY_DIR 覆盖）。"""
    settings = get_settings()
    base = getattr(settings, "MEMORY_DIR", "") or "memory"
    p = Path(base)
    if not p.is_absolute():
        # 相对路径锚定项目根（agent/ 的上一级）
        p = Path(__file__).resolve().parents[2] / p
    return p


def _index_file() -> Path:
    return _memory_dir() / "MEMORY.md"

_TOPIC_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def _safe_topic_path(topic: str) -> Path | None:
    """主题名 -> memory/ 内的安全路径；非法（逃逸/绝对路径/特殊字符）返回 None。"""
    # MEMORY 保留给索引文件 memory/MEMORY.md：topic 同名会覆盖索引本身
    # （Windows 文件系统大小写不敏感，须忽略大小写判定）
    if not topic or topic.upper() == "MEMORY" or not _TOPIC_NAME_RE.match(topic):
        return None
    p = (_memory_dir() / f"{topic}.md").resolve()
    # 双保险：resolve 后必须仍在 memory/ 目录内
    if _memory_dir().resolve() not in p.parents:
        return None
    return p


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except OSError as e:
        logger.warning("[longterm] 读取失败 %s：%s", path, e)
        return ""


def _list_topics() -> list[Path]:
    d = _memory_dir()
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.md") if p.name != "MEMORY.md")


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """拆出主题文件的 frontmatter（created/updated 日期）与正文。

    无 frontmatter 的历史文件返回 ({}, 原文)，兼容渐进迁移。
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        k, sep, v = line.partition(":")
        if not sep:
            continue
        k, v = k.strip(), v.strip()
        if k in ("created", "updated") and _DATE_VALUE_RE.match(v):
            meta[k] = v
    return meta, text[m.end():]


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _read_topic(path: Path) -> tuple[dict[str, str], str]:
    """读主题文件并拆分元数据（created/updated, 正文）。"""
    return _split_frontmatter(_read(path))


def _write_topic_file(path: Path, content: str, created: str | None = None) -> None:
    """写主题文件（自动维护 created/updated frontmatter，原子写）。"""
    today = _today()
    created = created or today
    body = f"---\ncreated: {created}\nupdated: {today}\n---\n{content.strip()}\n"
    _atomic_write(path, body)


def apply_longterm_edits(edits: list[dict]) -> list[dict]:
    """执行反思产生的自动记忆编辑（只允许写 memory/ 目录）。

    Args:
        edits: [{"topic": "user-prefs", "action": "append|update|create", "content": "..."}]

    Returns:
        实际应用的编辑列表（被安全闸/查重拒绝的编辑不返回，仅记日志）。
    """
    from agent.utils import text_coverage

    applied: list[dict] = []
    if not edits or not _is_enabled():
        return applied

    with _lock:
        d = _memory_dir()
        d.mkdir(parents=True, exist_ok=True)
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            topic = str(edit.get("topic", "")).strip()
            action = str(edit.get("action", "")).strip()
            content = str(edit.get("content", "")).strip()
            if not content or action not in ("append", "update", "create"):
                continue
            path = _safe_topic_path(topic)
            if path is None:
                logger.warning("[longterm] 拒绝非法主题名写入：%r", topic)
                continue

            meta: dict[str, str] = {}
            if action == "create" and path.exists():
                action = "append"  # create 遇到已存在 → 降级为 append
            if path.exists():
                meta, _existing = _read_topic(path)
            if action == "update" and not path.exists():
                action = "create"
            if action == "append" and not path.exists():
                action = "create"

            if action == "create":
                _write_topic_file(path, content)
            elif action == "append":
                _, existing_content = _read_topic(path)
                # 写入查重：新内容基本已被该主题覆盖 → 跳过（防重复纠正堆砌近重复行）
                if text_coverage(content, existing_content) >= _APPEND_DUP_THRESHOLD:
                    logger.info("[longterm] 追加内容与主题 %s 近重复，跳过", topic)
                    continue
                merged = (existing_content + "\n" + content) if existing_content else content
                _write_topic_file(path, merged, created=meta.get("created"))
            else:  # update
                _, existing_content = _read_topic(path)
                if existing_content and text_coverage(existing_content, content) < _UPDATE_REPLACE_THRESHOLD:
                    # 新内容未保留既有要点 → 降级为 append，不丢旧记忆
                    logger.info("[longterm] update 内容未保留主题 %s 既有要点，降级为 append", topic)
                    action = "append"
                    _write_topic_file(path, existing_content + "\n" + content, created=meta.get("created"))
                else:
                    _write_topic_file(path, content, created=meta.get("created"))

            _update_index_entry(topic, content)
            applied.append({"topic": topic, "action": action, "content": content})
            logger.info("[longterm] 自动记忆写入：topic=%s action=%s", topic, action)

    return applied


def _atomic_write(path: Path, content: str) -> None:
    """原子写：先写临时文件再 rename，避免半截文件。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _index_line_pattern(topic: str) -> re.Pattern[str]:
    """主题在索引中的行模式（兼容带/不带更新日期两种历史格式）。"""
    return re.compile(rf"^- {re.escape(topic)}(?:（[^）]*）)?:.*$", re.M)


def _update_index_entry(topic: str, content: str, updated: str | None = None) -> None:
    """维护 memory/MEMORY.md 索引行（带更新日期，不存在则创建索引骨架）。"""
    idx = _index_file()
    summary = re.sub(r"\s+", " ", content)[:_TOPIC_SUMMARY_LEN]
    line = f"- {topic}（更新 {updated or _today()}）: {summary}"
    existing = _read(idx)
    if not existing:
        _atomic_write(idx, "# Agent 自动记忆索引\n\n" + line + "\n")
        return
    # 替换或追加该主题行
    pattern = _index_line_pattern(topic)
    if pattern.search(existing):
        new_text = pattern.sub(line, existing)
    else:
        new_text = existing.rstrip() + "\n" + line + "\n"
    _atomic_write(idx, new_text)


def _is_enabled() -> bool:
    settings = get_settings()
    return bool(getattr(settings, "AUTO_MEMORY_ENABLED", True))


def get_auto_memory_overview() -> dict:
    """自动记忆概览（治理 API 用）：索引原文 + 主题文件清单（含时间戳与正文）。"""
    topics = []
    for f in _list_topics():
        meta, content = _read_topic(f)
        topics.append({
            "topic": f.stem,
            "created": meta.get("created", ""),
            "updated": meta.get("updated", ""),
            "content": content,
        })
    return {
        "index": _read(_index_file()),
        "topics": topics,
    }


def read_topic(topic: str) -> str | None:
    """读指定主题文件；主题名非法或不存在返回 None。"""
    p = _safe_topic_path(topic)
    if p is None or not p.exists():
        return None
    return _read(p)


def write_topic(topic: str, content: str) -> bool:
    """人工编辑/创建主题文件（治理 API 用），保留 created、刷新 updated，同步索引。"""
    p = _safe_topic_path(topic)
    if p is None:
        return False
    with _lock:
        _memory_dir().mkdir(parents=True, exist_ok=True)
        meta: dict[str, str] = {}
        if p.exists():
            meta, _ = _read_topic(p)
        _write_topic_file(p, content, created=meta.get("created"))
        _update_index_entry(topic, content)
    return True


def delete_topic(topic: str) -> bool:
    """删除主题文件并从索引移除（治理 API 用）。"""
    p = _safe_topic_path(topic)
    if p is None or not p.exists():
        return False
    with _lock:
        p.unlink()
        idx = _index_file()
        existing = _read(idx)
        if existing:
            pattern = re.compile(
                rf"^- {re.escape(topic)}(?:（[^）]*）)?:.*$\n?", re.M)
            _atomic_write(idx, pattern.sub("", existing))
    return True


def repair_index() -> int:
    """目录治理：为孤儿主题文件重建索引行 + 清理指向不存在文件的索引行。

    渐进式披露下索引是唯一入口：索引行与主题文件必须一致——
    文件缺行的补上（Curator 对账），行指向的文件已不存在的删掉
    （否则模型按索引调 read_memory_topic 会 404）。
    返回修复条数（补行 + 清行）。
    """
    fixed = 0
    with _lock:
        idx = _index_file()
        existing_idx = _read(idx)
        topic_names = {f.stem for f in _list_topics()}
        if existing_idx:
            lines = existing_idx.splitlines()
            kept: list[str] = []
            for line in lines:
                m = re.match(r"^- ([A-Za-z0-9][A-Za-z0-9_-]*)(?:（[^）]*）)?:", line)
                if m and m.group(1) not in topic_names:
                    fixed += 1          # 孤儿索引行：主题文件不存在 → 删除
                    continue
                kept.append(line)
            cleaned = "\n".join(kept).rstrip() + "\n"
            if cleaned != existing_idx:
                _atomic_write(idx, cleaned)
        for name in topic_names:
            if _index_line_pattern(name).search(_read(idx)):
                continue
            meta, content = _read_topic(_memory_dir() / f"{name}.md")
            _update_index_entry(name, content, updated=meta.get("updated"))
            fixed += 1
    return fixed
