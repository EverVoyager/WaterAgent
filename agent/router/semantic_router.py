"""基于 embedding 的语义路由核心实现。

设计参考：https://github.com/aurelio-labs/semantic-router
核心思路：
  1. 为每个意图定义若干示例 utterances
  2. 启动时预计算所有 utterances 的 embedding
  3. 用户输入也计算 embedding
  4. 余弦相似度匹配 + 阈值过滤
"""
import logging
import math
from typing import List

from pydantic import BaseModel, Field

from app.core.llm import get_llm_client, get_llm_config

logger = logging.getLogger(__name__)

# 默认 embedding 维度兜底（OpenAI text-embedding-3-small 是 1536，
# 阿里 DashScope text-embedding-v2 是 1536，v3 是 1024）
DEFAULT_EMBEDDING_MODEL = "text-embedding-v3"


class SemanticRoute(BaseModel):
    """单个语义路由。"""

    name: str = Field(..., description="意图名称")
    utterances: List[str] = Field(..., description="示例话术列表")


class RouteDecision(BaseModel):
    """路由决策结果。"""

    route_name: str = Field("", description="匹配到的意图名；未匹配为空")
    score: float = Field(0.0, description="最高相似度分数")
    matched_utterance: str = Field("", description="命中的示例话术")
    fallback_reason: str = Field("", description="未匹配或降级时的原因")


class SemanticRouter:
    """语义路由器：基于 embedding 余弦相似度做意图匹配。

    使用方式：
        router = SemanticRouter(routes=[...])
        decision = router("用户输入")
        if decision.route_name:
            ...
    """

    def __init__(
        self,
        routes: List[SemanticRoute],
        threshold: float = 0.55,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.routes = routes
        self.threshold = threshold
        self.embedding_model = embedding_model

        # 展平所有 utterances，记录其所属路由索引
        self._utterances: List[str] = []
        self._route_idx: List[int] = []
        for idx, r in enumerate(routes):
            for utt in r.utterances:
                self._utterances.append(utt)
                self._route_idx.append(idx)

        # 预计算 utterance embedding
        self._embeddings: List[List[float]] = []
        self._ready = False
        try:
            self._embeddings = self._encode(self._utterances)
            self._ready = True
            logger.info(
                "[semantic-router] ready: %d routes, %d utterances, model=%s",
                len(routes), len(self._utterances), embedding_model,
            )
        except Exception as e:
            logger.warning(
                "[semantic-router] embedding precompute failed: %s — will fallback to rules",
                e,
            )

    @property
    def ready(self) -> bool:
        """是否就绪（embedding 已预计算）。"""
        return self._ready and bool(self._embeddings)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """调用 embedding API。"""
        if not texts:
            return []
        settings = get_llm_config()
        client = get_llm_client()
        # DashScope embedding 接口单次最多 10 条文本
        batch_size = 10
        all_vecs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(
                model=settings.get("embedding_model") or self.embedding_model,
                input=batch,
            )
            # 按 index 排序保证顺序
            sorted_data = sorted(resp.data, key=lambda x: x.index)
            all_vecs.extend([d.embedding for d in sorted_data])
        return all_vecs

    def __call__(self, query: str) -> RouteDecision:
        """对用户输入做意图匹配。

        Returns:
            RouteDecision，未匹配时 route_name 为空字符串
        """
        if not self.ready:
            return RouteDecision(fallback_reason="semantic_router_not_ready")

        try:
            q_emb = self._encode([query])[0]
        except Exception as e:
            logger.warning("[semantic-router] query embedding failed: %s", e)
            return RouteDecision(fallback_reason=f"embedding_failed: {e}")

        # 计算余弦相似度
        best_score = -1.0
        best_idx = -1
        for i, utt_emb in enumerate(self._embeddings):
            score = _cosine(q_emb, utt_emb)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0 or best_score < self.threshold:
            return RouteDecision(
                score=best_score if best_idx >= 0 else 0.0,
                fallback_reason=f"below_threshold_{best_score:.3f}<{self.threshold}",
            )

        route = self.routes[self._route_idx[best_idx]]
        return RouteDecision(
            route_name=route.name,
            score=best_score,
            matched_utterance=self._utterances[best_idx],
        )


def _cosine(a: List[float], b: List[float]) -> float:
    """余弦相似度。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
