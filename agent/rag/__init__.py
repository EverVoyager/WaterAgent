"""RAG 法规检索模块。

提供文档加载、切分、向量化、检索能力，支撑 search_regulation 工具。
"""
from agent.rag.document_loader import (
    RegulationChunk,
    load_regulation_files,
    split_markdown_to_chunks,
)
from agent.rag.embedding import embed_query, embed_texts
from agent.rag.vector_store import (
    build_and_persist_index,
    is_index_ready,
    load_vector_store,
    search_regulations,
)

__all__ = [
    "RegulationChunk",
    "load_regulation_files",
    "split_markdown_to_chunks",
    "embed_texts",
    "embed_query",
    "build_and_persist_index",
    "load_vector_store",
    "is_index_ready",
    "search_regulations",
]
