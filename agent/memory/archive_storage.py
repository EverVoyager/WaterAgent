"""任务段文件存储：原子写入、幂等更新、安全读取与过期清理。"""
import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.memory.session_segments import Round, Segment
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_WRITE_LOCK = threading.RLock()


# ============ 归档文件（MD + JSON meta） ============

def _archive_dir() -> Path:
    settings = get_settings()
    d = Path(settings.SESSION_ARCHIVE_DIR)
    if not d.is_absolute():
        # 相对路径锚定项目根（与 longterm 的 memory/ 目录惯例一致）
        d = Path(__file__).resolve().parents[2] / d
    return d


def _seg_path(first_fp: str) -> Path:
    return _archive_dir() / f"{first_fp}.md"


def _load_meta(first_fp: str) -> dict[str, Any] | None:
    """读段文件 meta（JSON 单行 frontmatter）。文件不存在返回 None。"""
    p = _seg_path(first_fp)
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8")
        m = re.match(r"^---\nmeta: (.*)\n---\n", text, re.DOTALL)
        if not m:
            return None
        return json.loads(m.group(1))
    except Exception as e:
        logger.warning("[session-archive] 段文件 meta 解析失败 fp=%s：%s", first_fp, e)
        return None


def _read_body(first_fp: str) -> str:
    p = _seg_path(first_fp)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8")
    m = re.match(r"^---\nmeta: .*\n---\n", text, re.DOTALL)
    return text[m.end():] if m else text


def _save_segment(first_fp: str, meta: dict[str, Any], body: str) -> None:
    """原子写段文件（tmp + replace，与 longterm 的 _atomic_write 同惯例）。"""
    d = _archive_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = _seg_path(first_fp)
    tmp = p.with_suffix(".tmp")
    content = f"---\nmeta: {json.dumps(meta, ensure_ascii=False)}\n---\n{body}"
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(p)


def upsert_round(
    first_fp: str,
    rounds: list[Round],
    round_: Round,
    tool_data: str = "",
) -> None:
    """把一轮写入段文件（幂等：按轮指纹存在则替换——收尾归档用含工具
    数据的版本覆盖入口归档的纯文本版本）。

    Args:
        rounds: 该段当前已知的全部轮次（用于维护 meta 的轮指纹/queries 序列）
        round_: 待写入轮
        tool_data: 工具轨迹文本（收尾归档提供；入口归档为空）
    """
    with _WRITE_LOCK:
        meta = _load_meta(first_fp) or {
            "first_fp": first_fp,
            "round_fps": [],
            "queries": [],
            "summaries": [],   # [{covered: n, text: "..."}]，冻结后只追加
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        body = _read_body(first_fp)
        section = _format_round_section(round_, tool_data)
        pattern = re.compile(rf"^### 轮 {round_.fp}\n.*?(?=^### 轮 |\Z)", re.DOTALL | re.MULTILINE)
        if pattern.search(body):
            body = pattern.sub(section, body)
        else:
            body = body + section
        # 维护轮序列（以调用方给定的 rounds 为准，幂等去重）
        known = list(meta.get("round_fps", []))
        queries = list(meta.get("queries", []))
        if round_.fp not in known:
            known.append(round_.fp)
            queries.append(round_.query)
        meta["round_fps"] = known
        meta["queries"] = queries
        try:
            _save_segment(first_fp, meta, body)
        except Exception as e:
            logger.warning("[session-archive] 段写入失败 fp=%s：%s", first_fp, e)


def _format_round_section(round_: Round, tool_data: str) -> str:
    lines = [f"### 轮 {round_.fp}", f"用户：{round_.query}"]
    if tool_data:
        lines.append(f"工具轨迹：\n{tool_data}")
    lines.append(f"回答：{round_.answer}")
    return "\n".join(lines) + "\n\n"


def archive_rounds(segments: list[Segment], tool_data_by_fp: dict[str, str] | None = None) -> None:
    """入口归档：把分段结果写入段文件（已存在且轮已归档则跳过）。"""
    settings = get_settings()
    if not settings.SESSION_ARCHIVE_ENABLED:
        return
    tool_data_by_fp = tool_data_by_fp or {}
    for seg in segments:
        meta = _load_meta(seg.first_fp)
        for r in seg.rounds:
            if meta and r.fp in meta.get("round_fps", []):
                continue  # 幂等：该轮已归档
            upsert_round(seg.first_fp, seg.rounds, r, tool_data_by_fp.get(r.fp, ""))

_ARCHIVE_FILENAME_RE = re.compile(r"^[0-9a-f]{16}\.md$")

# 单次读取上限（字符）：一段含多轮 + 工具轨迹，防止一次读取撑爆上下文
ARCHIVE_READ_MAX_CHARS = 8000


def _archive_filename(first_fp: str) -> str:
    """段存档文件名（与 _seg_path 一致）：<16位指纹>.md，读取工具按名取件。"""
    return f"{first_fp}.md"


def read_archive_file(filename: str, max_chars: int = ARCHIVE_READ_MAX_CHARS) -> dict[str, Any]:
    """读取段存档全文（read_session_archive 工具的底层实现）。

    文件名严格校验（16 位十六进制指纹 + .md，即 _round_fp 产出的命名），
    杜绝路径穿越；meta frontmatter 剥离，只返回正文；超长截断并标记。

    Raises:
        ValueError: 文件名不合法
        FileNotFoundError: 存档不存在（归档未跑/已清理）
    """
    if not _ARCHIVE_FILENAME_RE.match(filename or ""):
        raise ValueError(
            "存档文件名不合法：应为 16 位十六进制指纹.md，"
            "来自历史任务摘要末尾的 [存档] 行"
        )
    first_fp = filename[:-3]
    p = _seg_path(first_fp)
    if not p.exists():
        raise FileNotFoundError(f"存档文件不存在：{filename}（可能已被过期清理）")
    body = _read_body(first_fp).strip()
    truncated = len(body) > max_chars
    return {
        "file": filename,
        "content": body[:max_chars],
        "truncated": truncated,
        "total_chars": len(body),
        "read_at": datetime.now().isoformat(timespec="seconds"),
    }


def cleanup_archive() -> int:
    """删除超龄归档段（Curator 周期调用）。返回删除的文件数。"""
    settings = get_settings()
    d = _archive_dir()
    if not d.exists():
        return 0
    cutoff = datetime.now().timestamp() - settings.SESSION_ARCHIVE_MAX_AGE_DAYS * 86400
    removed = 0
    for p in d.glob("*.md"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            continue
    if removed:
        logger.info("[session-archive] 清理超龄归档 %d 个（> %d 天）",
                    removed, settings.SESSION_ARCHIVE_MAX_AGE_DAYS)
    return removed
