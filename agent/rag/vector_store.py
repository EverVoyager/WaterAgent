"""Qdrant 向量库：构建 / 加载 / 检索。

设计：
- Collection：water_regulations（可配置）
- 距离度量：Cosine（Qdrant 内部做归一化，无需手动 L2 normalize）
- 索引类型：HNSW（Qdrant 默认，适合中小规模 + 高召回）
- Payload：title / doc_type / chapter / article / text / source_file 等 metadata
- 持久化：Qdrant 服务自身负责（data 目录在 qdrant.exe 旁的 storage/）

数据流：
    build_and_persist_index(chunks)
        → 创建/重建 collection → upsert 向量 + payload

    is_index_ready()
        → 检查 Qdrant 服务可达 + collection 存在 + points 数 > 0

    search_regulations(query, top_k)
        → embed_query → qdrant.search → 返回 [{title, article, content, score}]

接口与 FAISS 版完全一致，调用方无需改动。
"""
import logging
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from agent.rag.document_loader import RegulationChunk
from agent.rag.embedding import embed_query, embed_texts
from app.core.llm import get_qdrant_client, get_qdrant_config

logger = logging.getLogger(__name__)

# 兼容旧引用（保留符号，但不再使用文件目录路径）
FAISS_INDEX_DIR = "data/processed/faiss_index"


# ====== 客户端 ======

def _get_client() -> QdrantClient:
    """获取 Qdrant 客户端单例。"""
    return get_qdrant_client()


def _collection_name() -> str:
    return get_qdrant_config()["collection"]


def _vector_size() -> int:
    return get_qdrant_config()["vector_size"]


# ====== 构建 & 持久化 ======

def build_and_persist_index(
    chunks: List[RegulationChunk],
    output_dir: str = FAISS_INDEX_DIR,  # 保留参数以兼容旧签名，实际忽略
) -> int:
    """构建 Qdrant collection：创建/重建 → 批量 upsert 向量 + payload。

    Args:
        chunks: 已切分的法规 chunks
        output_dir: 兼容旧签名的参数，Qdrant 版忽略

    Returns:
        索引中向量数量
    """
    if not chunks:
        logger.warning("[vector_store] chunks 为空，跳过构建")
        return 0

    client = _get_client()
    collection = _collection_name()
    dim = _vector_size()

    # embedding 全部 chunks
    texts = [c.text for c in chunks]
    logger.info("[vector_store] embedding %d chunks ...", len(texts))
    vectors = embed_texts(texts)
    if vectors is None:
        raise RuntimeError("Embedding 失败，无法构建索引")

    # 若 collection 已存在则先删除（重建）
    try:
        client.delete_collection(collection_name=collection)
        logger.info("[vector_store] 已删除旧 collection: %s", collection)
    except UnexpectedResponse:
        pass  # collection 不存在，忽略

    # 创建 collection（Cosine 距离）
    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(
            size=dim,
            distance=qmodels.Distance.COSINE,
        ),
    )
    logger.info("[vector_store] 已创建 collection: %s (dim=%d)", collection, dim)

    # 批量 upsert（每批 64 条）
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        payload = {
            "id": i,
            "text": chunk.text,
            **chunk.to_metadata(),
        }
        points.append(qmodels.PointStruct(
            id=i,
            vector=vec.tolist(),
            payload=payload,
        ))

    batch_size = 64
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection, points=batch)
        logger.debug("[vector_store] upsert %d-%d", i, i + len(batch))

    # 为 payload 字段创建索引（提升过滤检索性能）
    for field in ("title", "doc_type", "chapter", "source_file"):
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            logger.debug("[vector_store] 创建 payload 索引失败 %s: %s", field, e)

    total = client.count(collection_name=collection, exact=True).count
    logger.info("[vector_store] Qdrant 索引构建完成: %d points", total)
    return total


# ====== 加载 & 就绪检查 ======

def load_vector_store(
    index_dir: str = FAISS_INDEX_DIR,  # 兼容旧签名，忽略
):
    """兼容旧接口：Qdrant 不需要显式加载，直接返回 client + collection 名。

    Returns:
        (client, collection_name) 或 None（服务不可达 / collection 不存在）
    """
    if not is_index_ready():
        return None
    return _get_client(), _collection_name()


def is_index_ready(index_dir: str = FAISS_INDEX_DIR) -> bool:
    """检查 Qdrant 服务可达 + collection 存在 + points 数 > 0。"""
    try:
        client = _get_client()
        collection = _collection_name()
        cols = client.get_collections().collections
        names = [c.name for c in cols]
        if collection not in names:
            return False
        count = client.count(collection_name=collection, exact=True).count
        return count > 0
    except Exception as e:
        logger.warning("[vector_store] Qdrant 不可用: %s", e)
        return False


# ====== 检索 ======

def search_regulations(
    query: str,
    top_k: int = 3,
    index_dir: str = FAISS_INDEX_DIR,  # 兼容旧签名，忽略
    min_score: float = 0.3,
) -> List[Dict[str, Any]]:
    """检索法规。

    Args:
        query: 检索关键词或自然语言问题
        top_k: 返回前 K 条
        index_dir: 兼容旧签名，忽略
        min_score: 最低相似度阈值（过滤低质量命中）

    Returns:
        [{"title", "article", "chapter", "content", "score", "doc_type", "source_file"}, ...]
    """
    if not is_index_ready():
        logger.warning("[vector_store] Qdrant 索引未就绪，无法检索")
        return []

    query_vec = embed_query(query)
    if query_vec is None:
        logger.warning("[vector_store] query embedding 失败")
        return []

    client = _get_client()
    collection = _collection_name()

    # qdrant-client 1.10+ 用 query_points 替代废弃的 search
    response = client.query_points(
        collection_name=collection,
        query=query_vec.tolist(),
        limit=top_k,
        with_payload=True,
    )

    results: List[Dict[str, Any]] = []
    for scored in response.points:
        score = float(scored.score)
        if score < min_score:
            continue
        payload = scored.payload or {}
        results.append({
            "title": payload.get("title", ""),
            "article": payload.get("article", ""),
            "chapter": payload.get("chapter", ""),
            "content": payload.get("text", ""),
            "doc_type": payload.get("doc_type", ""),
            "source_file": payload.get("source_file", ""),
            "score": round(score, 4),
        })
    return results
