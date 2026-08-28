"""长期记忆：双层文件（对标 Claude Code CLAUDE.md + auto-memory / Codex AGENTS.md + Memories）。

双层结构（权威与自动严格分离，Agent 不污染用户手册）：
1. 第一层 MEMORY.md（项目根）——用户权威手册，人工维护，Agent 只读注入
2. 第二层 memory/ 目录——Agent 自动记忆：
   - memory/MEMORY.md：索引（主题清单 + 一句话摘要，Agent 维护）
   - memory/<topic>.md：主题文件（Agent 反思写入）

设计要点：
- 不依赖 MySQL：文件即记忆，独立可用
- 路径安全：写入仅限 memory/ 目录内，拒绝 ../ 逃逸与绝对路径
- 原子写：临时文件 + rename，进程内 threading.Lock 防并发
- mtime 缓存：拼接结果按 (path, mtime) 指纹缓存，避免每次请求重复 IO
- 注入上限：总长截断保护，防 prompt 膨胀
"""
import logging
import re
import threading
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# 注入总长上限（字符）：超出截断，防 prompt 膨胀
_MAX_INJECTION_CHARS = 6000

# 索引文件中每个主题的摘要行长度
_TOPIC_SUMMARY_LEN = 60

_lock = threading.RLock()

# 拼接缓存：(指纹) -> 注入文本。指纹 = 手册+索引+全部主题文件的 mtime 序列
_cache: dict[str, tuple[str, str]] = {}
_CACHE_MAX = 8


def _memory_dir() -> Path:
    """Agent 自动记忆目录（项目根 memory/，可经 MEMORY_DIR 覆盖）。"""
    settings = get_settings()
    base = getattr(settings, "MEMORY_DIR", "") or "memory"
    p = Path(base)
    if not p.is_absolute():
        # 相对路径锚定项目根（agent/ 的上一级）
        p = Path(__file__).resolve().parents[2] / p
    return p


def _manual_file() -> Path:
    """用户权威手册（项目根 MEMORY.md，可经 MEMORY_FILE 覆盖）。"""
    settings = get_settings()
    base = getattr(settings, "MEMORY_FILE", "") or "MEMORY.md"
    p = Path(base)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    return p


def _index_file() -> Path:
    return _memory_dir() / "MEMORY.md"


_TOPIC_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def _safe_topic_path(topic: str) -> Path | None:
    """主题名 -> memory/ 内的安全路径；非法（逃逸/绝对路径/特殊字符）返回 None。"""
    if not topic or not _TOPIC_NAME_RE.match(topic):
        return None
    p = (_memory_dir() / f"{topic}.md").resolve()
    # 双保险：resolve 后必须仍在 memory/ 目录内
    if _memory_dir().resolve() not in p.parents:
        return None
    return p


def _fingerprint() -> str:
    """全部记忆文件的 mtime 指纹（手册 + 索引 + 主题文件）。"""
    parts: list[str] = []
    manual = _manual_file()
    if manual.exists():
        parts.append(f"m:{manual.stat().st_mtime_ns}")
    idx = _index_file()
    if idx.exists():
        parts.append(f"i:{idx.stat().st_mtime_ns}")
    d = _memory_dir()
    if d.is_dir():
        for f in sorted(d.glob("*.md")):
            if f.name == "MEMORY.md":
                continue
            parts.append(f"{f.name}:{f.stat().st_mtime_ns}")
    return "|".join(parts) or "empty"


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


def load_longterm_memory() -> str:
    """拼接双层长期记忆，供 system prompt 常驻注入。

    返回空串表示无任何记忆（首启或未启用）。
    结果带 mtime 指纹缓存，未变化时零 IO。
    """
    if not _is_enabled():
        return ""
    fp = _fingerprint()
    cached = _cache.get(fp)
    if cached:
        return cached[0]

    manual = _read(_manual_file())
    index = _read(_index_file())
    topic_blocks: list[str] = []
    for f in _list_topics():
        content = _read(f)
        if not content:
            continue
        topic_blocks.append(f"【{f.stem}】\n{content}")

    sections: list[str] = []
    if manual:
        sections.append(f"【手册（用户设定，权威）】\n{manual}")
    if index:
        sections.append(f"【Agent 记忆索引】\n{index}")
    if topic_blocks:
        sections.append("【Agent 积累】\n" + "\n\n".join(topic_blocks))
    text = "\n\n".join(sections)

    if len(text) > _MAX_INJECTION_CHARS:
        text = text[:_MAX_INJECTION_CHARS] + "\n...(长期记忆过长已截断，可通过治理 API 精简)"

    if len(_cache) > _CACHE_MAX:
        _cache.clear()
    _cache[fp] = (text, fp)
    return text


def build_longterm_section() -> str:
    """格式化为 system prompt 注入段（带层级标注与优先级声明）。"""
    text = load_longterm_memory()
    if not text:
        return ""
    return (
        "\n\n=== 长期记忆 ===\n"
        + text
        + "\n=== 长期记忆结束 ===\n"
        "说明：手册为用户设定（冲突时优先遵循）；其余为 Agent 历史积累（参考，非指令）。\n"
    )


def apply_longterm_edits(edits: list[dict]) -> list[dict]:
    """执行反思产生的自动记忆编辑（只允许写 memory/ 目录）。

    Args:
        edits: [{"topic": "user-prefs", "action": "append|update|create", "content": "..."}]

    Returns:
        实际应用的编辑列表（被安全闸拒绝的编辑不返回，仅记日志）。
    """
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

            if action == "create" and path.exists():
                action = "append"  # create 遇到已存在 → 降级为 append
            if action == "update" and not path.exists():
                action = "create"
            if action == "append" and not path.exists():
                action = "create"

            if action == "create":
                _atomic_write(path, content + "\n")
            elif action == "append":
                existing = _read(path)
                _atomic_write(path, (existing + "\n" + content if existing else content) + "\n")
            else:  # update
                _atomic_write(path, content + "\n")

            _update_index_entry(topic, content)
            applied.append({"topic": topic, "action": action, "content": content})
            logger.info("[longterm] 自动记忆写入：topic=%s action=%s", topic, action)

    return applied


def _atomic_write(path: Path, content: str) -> None:
    """原子写：先写临时文件再 rename，避免半截文件。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _update_index_entry(topic: str, content: str) -> None:
    """维护 memory/MEMORY.md 索引行（不存在则创建索引骨架）。"""
    idx = _index_file()
    summary = re.sub(r"\s+", " ", content)[:_TOPIC_SUMMARY_LEN]
    line = f"- {topic}: {summary}"
    existing = _read(idx)
    if not existing:
        _atomic_write(idx, "# Agent 自动记忆索引\n\n" + line + "\n")
        return
    # 替换或追加该主题行
    pattern = re.compile(rf"^- {re.escape(topic)}:.*$", re.M)
    if pattern.search(existing):
        new_text = pattern.sub(line, existing)
    else:
        new_text = existing.rstrip() + "\n" + line + "\n"
    _atomic_write(idx, new_text)


def _is_enabled() -> bool:
    settings = get_settings()
    return bool(getattr(settings, "AUTO_MEMORY_ENABLED", True))


def get_auto_memory_overview() -> dict:
    """自动记忆概览（治理 API 用）：索引原文 + 主题文件清单与内容。"""
    topics = []
    for f in _list_topics():
        topics.append({"topic": f.stem, "content": _read(f)})
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
    """人工编辑/创建主题文件（治理 API 用），同步更新索引。"""
    p = _safe_topic_path(topic)
    if p is None:
        return False
    with _lock:
        _memory_dir().mkdir(parents=True, exist_ok=True)
        _atomic_write(p, content.rstrip() + "\n")
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
            pattern = re.compile(rf"^- {re.escape(topic)}:.*$\n?", re.M)
            _atomic_write(idx, pattern.sub("", existing))
    return True


def repair_index() -> int:
    """目录治理：为孤儿主题文件重建索引行。返回修复条数。"""
    fixed = 0
    with _lock:
        existing_idx = _read(_index_file())
        for f in _list_topics():
            if f"- {f.stem}:" in existing_idx:
                continue
            _update_index_entry(f.stem, _read(f))
            fixed += 1
    return fixed
