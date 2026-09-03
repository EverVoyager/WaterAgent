"""DashScope Embedding 封装。

使用独立的 embedding 客户端（EMBEDDING_API_KEY / EMBEDDING_BASE_URL，
留空回退推理 LLM_* 配置），调用 text-embedding-v3（1024 维）。
所有向量都做 L2 归一化，便于用内积（IndexFlatIP）直接得到余弦相似度。
"""
import logging

import numpy as np

from app.core.llm import get_embedding_client, get_llm_config

logger = logging.getLogger(__name__)

# DashScope text-embedding-v3 输出维度
EMBEDDING_DIM = 1024
# DashScope embedding 接口单次最多 10 条文本
BATCH_SIZE = 10


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """L2 归一化，使内积 = 余弦相似度。"""
    if vecs.size == 0:
        return vecs
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """调用一次 embedding API（最多 10 条）。"""
    settings = get_llm_config()
    client = get_embedding_client()
    resp = client.embeddings.create(
        model=settings.get("embedding_model"),
        input=texts,
    )
    # DashScope 返回的 data 可能乱序，按 index 排序
    sorted_data = sorted(resp.data, key=lambda x: x.index)
    return [d.embedding for d in sorted_data]


def embed_texts(texts: list[str]) -> np.ndarray | None:
    """批量 embedding，返回 L2 归一化后的 numpy 数组 (N, 1024)。

    失败时返回 None（由调用方决定降级策略）。
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    all_vecs: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        try:
            vecs = _embed_batch(batch)
            all_vecs.extend(vecs)
        except Exception as e:
            logger.error("[embedding] batch %d-%d 失败: %s", i, i + len(batch), e)
            return None

    arr = np.array(all_vecs, dtype=np.float32)
    return _l2_normalize(arr)


def embed_query(query: str) -> np.ndarray | None:
    """单条 query embedding，返回 L2 归一化后的 (1024,) 向量。"""
    arr = embed_texts([query])
    if arr is None or arr.shape[0] == 0:
        return None
    return arr[0]
