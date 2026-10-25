"""会话任务分段：轮次模型、query 向量缓存与稳定语义边界。"""
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# 轮 query embedding 进程内缓存（query 指纹 -> 向量），
# 使入口分段与收尾分段的重算成本 ≈ 每轮一次 embedding
_EMB_CACHE: dict[str, list[float]] = {}

_EMB_CACHE_MAX = 512

_EMB_LOCK = threading.Lock()


# ============ 数据结构 ============

@dataclass
class Round:
    """一轮对话（user query + assistant answer）。"""

    query: str
    answer: str
    fp: str = ""

    def __post_init__(self):
        if not self.fp:
            self.fp = _round_fp(self.query, self.answer)


@dataclass
class Segment:
    """任务段：连续相关的轮次集合。"""

    rounds: list[Round] = field(default_factory=list)

    @property
    def first_fp(self) -> str:
        return self.rounds[0].fp if self.rounds else ""

    @property
    def queries_text(self) -> str:
        """段匹配文本：段内全部 user query 拼接（主题代表性最强）。"""
        return "；".join(r.query for r in self.rounds)


def _round_fp(query: str, answer: str) -> str:
    raw = f"{query}|{answer}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def extract_rounds(history: list[dict[str, Any]]) -> list[Round]:
    """从 history 提取轮次（user 消息 + 其后的 assistant 消息配对）。

    system 消息跳过；user 后无 assistant 时按空答案处理（防御非对齐输入）。
    """
    rounds: list[Round] = []
    pending_query: str | None = None
    for m in history or []:
        role = m.get("role", "")
        content = str(m.get("content", "") or "")
        if role == "user":
            if pending_query is not None:
                rounds.append(Round(query=pending_query, answer=""))
            pending_query = content
        elif role == "assistant" and pending_query is not None:
            rounds.append(Round(query=pending_query, answer=content))
            pending_query = None
    if pending_query is not None:
        rounds.append(Round(query=pending_query, answer=""))
    return rounds


# ============ 分段（语义边界，因果稳定） ============

def _embed_queries(queries: list[str]) -> list[list[float] | None]:
    """批量计算 query embedding（带进程内缓存）。失败对应项为 None。"""
    from agent.rag.embedding import embed_texts

    cached: dict[int, list[float]] = {}
    missing: list[tuple[int, str]] = []
    for i, q in enumerate(queries):
        fp = hashlib.md5(q.encode("utf-8")).hexdigest()
        with _EMB_LOCK:
            vec = _EMB_CACHE.get(fp)
        if vec is not None:
            cached[i] = vec
        else:
            missing.append((i, fp, q))

    if missing:
        try:
            vecs = embed_texts([q for _, _, q in missing])
            if vecs is not None and len(vecs) == len(missing):
                for (i, fp, _), vec in zip(missing, vecs, strict=False):
                    cached[i] = list(map(float, vec))
                    with _EMB_LOCK:
                        if len(_EMB_CACHE) >= _EMB_CACHE_MAX:
                            _EMB_CACHE.clear()
                        _EMB_CACHE[fp] = cached[i]
            else:
                logger.debug("[session-archive] query embedding 批量返回异常")
        except Exception as e:
            logger.debug("[session-archive] query embedding 失败（分段降级单段）：%s", e)

    return [cached.get(i) for i in range(len(queries))]


def segment_rounds(rounds: list[Round]) -> list[Segment]:
    """把轮次序列切成任务段。

    相邻轮 query embedding 余弦 < 阈值 → 开新段。边界只依赖已有轮次文本
    （历史事实），新轮次只可能追加新边界、不会改动旧边界（KV Cache
    不变量 3）。embedding 不可用时降级为单段。
    """
    if not rounds:
        return []
    settings = get_settings()
    threshold = settings.SESSION_SEGMENT_SIM_THRESHOLD
    if not settings.SESSION_ARCHIVE_ENABLED or len(rounds) == 1:
        return [Segment(rounds=list(rounds))]

    vecs = _embed_queries([r.query for r in rounds])
    if any(v is None for v in vecs):
        # 任一向量缺失（embedding 不可用）→ 单段降级，行为接近原合并摘要
        return [Segment(rounds=list(rounds))]

    segments = [Segment(rounds=[rounds[0]])]
    for _prev, cur, pv, cv in zip(rounds, rounds[1:], vecs, vecs[1:], strict=False):
        sim = _cosine(pv, cv)
        if sim < threshold:
            segments.append(Segment(rounds=[cur]))
        else:
            segments[-1].rounds.append(cur)
    return segments


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（embedding 已 L2 归一化，内积即余弦；此处仍防御性归一）。"""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(y * y for y in b) ** 0.5 or 1.0
    return dot / (na * nb)
