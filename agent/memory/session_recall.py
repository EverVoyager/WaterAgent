"""可选历史段召回：向量匹配与关键词降级。"""
import logging
from typing import Any

from agent.memory.archive_storage import _read_body
from agent.memory.session_segments import (
    Segment,
    _cosine,
    _embed_queries,
    extract_rounds,
    segment_rounds,
)
from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ============ 按需还原·向量匹配（可选回退，默认关闭） ============
#
# 主路径是"摘要标注存档文件名 + read_session_archive 工具按需读取"；
# 以下 query-段向量匹配注入由 SESSION_RECALL_ON_QUERY 控制（对照实验用）：
# 段数增多后每请求比较成本线性增长，且命中即整段注入较费 token。

def recall_relevant_segments(
    query: str,
    history: list[dict[str, Any]],
    keep_recent_rounds: int,
) -> str:
    """匹配还原：当前 query vs 早段（窗口外）匹配向量，命中返回段全文文本。

    末尾注入（KV Cache 友好）；embedding 不可用/无命中返回空串。
    """
    settings = get_settings()
    if not settings.SESSION_ARCHIVE_ENABLED:
        return ""
    rounds = extract_rounds(history)
    segments = segment_rounds(rounds)
    keep_msgs = keep_recent_rounds * 2
    window = history[len(history) - keep_msgs:] if keep_msgs > 0 else []
    window_rounds = {r.fp for r in extract_rounds(window)}

    early = [seg for seg in segments if any(r.fp not in window_rounds for r in seg.rounds)]
    if not early:
        return ""

    q_vec = _embed_queries([query])[0]
    if q_vec is None:
        # query 向量都算不出（embedding 不可用）→ 对全部早段关键词降级
        return _recall_by_keyword(query, early)

    targets = [(seg, v) for seg in early if (v := _segment_match_vec(seg)) is not None]
    if not targets:
        return _recall_by_keyword(query, early)

    scored = sorted(
        ((seg, _cosine(q_vec, vec)) for seg, vec in targets),
        key=lambda x: x[1], reverse=True,
    )
    recalled: list[str] = []
    for seg, score in scored[: settings.SESSION_ARCHIVE_TOP_K]:
        if score < settings.SESSION_ARCHIVE_MATCH_THRESHOLD:
            continue
        body = _read_body(seg.first_fp).strip()
        if not body:
            body = "\n".join(f"用户：{r.query}\n回答：{r.answer}" for r in seg.rounds)
        recalled.append(body)
    if not recalled:
        return ""
    return ("【相关历史任务还原（系统按相关性加载的往轮记录，含工具数据，参考）】\n"
            + "\n\n".join(recalled))


def _segment_match_vec(seg: Segment) -> list[float] | None:
    """段匹配向量：embedding(段内 query 拼接)。段短时直接现算（有缓存）。"""
    return _embed_queries([seg.queries_text])[0]


def _recall_by_keyword(query: str, segments: list[Segment]) -> str:
    """embedding 不可用时的关键词降级：query 字符与段 query 的重合命中。"""
    chars = {c for c in query if not c.isspace()}
    if not chars:
        return ""
    best: tuple[Segment, float] | None = None
    for seg in segments:
        seg_chars = {c for c in seg.queries_text if not c.isspace()}
        overlap = len(chars & seg_chars) / len(chars)
        if best is None or overlap > best[1]:
            best = (seg, overlap)
    if best and best[1] >= 0.3:
        seg = best[0]
        body = _read_body(seg.first_fp).strip()
        return ("【相关历史任务还原（关键词降级匹配）】\n" + body) if body else ""
    return ""
