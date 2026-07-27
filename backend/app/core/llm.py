"""LLM 客户端封装：基于 OpenAI 兼容接口。"""
from functools import lru_cache

import httpx
from openai import OpenAI
from qdrant_client import QdrantClient

from app.core.config import get_settings


# 不同节点的 LLM 调用超时配置（秒）
# 参考 agent-service-toolkit 的分级超时策略
# 调整：read 超时统一延长，避免网络抖动或 LLM 推理慢导致超时
LLM_TIMEOUTS = {
    "default": httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
    "planner": httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0),
    "reflector": httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0),
    "synthesizer": httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0),
    "chat": httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
    "embedding": httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
}


@lru_cache
def get_llm_client() -> OpenAI:
    """单例 OpenAI 客户端。默认 60s 读超时，调用方可通过 with_options 临时覆盖。"""
    settings = get_settings()
    return OpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        timeout=LLM_TIMEOUTS["default"],
        max_retries=0,  # Agent 层自己处理重试，避免 SDK 默认 2 次重试放大延迟
    )


def get_llm_config() -> dict:
    """返回 LLM 调用所需的配置。"""
    settings = get_settings()
    return {
        "model": settings.LLM_MODEL,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "max_tool_rounds": settings.LLM_MAX_TOOL_ROUNDS,
        "embedding_model": settings.LLM_EMBEDDING_MODEL,
    }


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """单例 Qdrant 客户端（连接本地 Qdrant 服务）。"""
    settings = get_settings()
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=10)


def get_qdrant_config() -> dict:
    """返回 Qdrant 检索所需的配置。"""
    settings = get_settings()
    return {
        "host": settings.QDRANT_HOST,
        "port": settings.QDRANT_PORT,
        "collection": settings.QDRANT_COLLECTION,
        "vector_size": settings.QDRANT_VECTOR_SIZE,
    }


def get_default_system_prompt() -> str:
    """默认系统提示词。"""
    return (
        "你是黄河吕梁段防汛预警智能体。你的职责是根据用户问题，"
        "调用合适的工具获取天气、水情、径流预测、GIS 地形、法规政策等信息，"
        "并基于工具返回的数据进行综合研判，输出预警等级与具体应急预案。\n\n"
        "规则：\n"
        "1. 一次可以调用多个工具，但每次只调用最相关的 1-3 个；\n"
        "2. 工具调用失败时，应说明原因并基于已有信息进行研判；\n"
        "3. 最终回答必须包含：预警等级（Ⅰ/Ⅱ/Ⅲ/Ⅳ）、研判依据、具体应急措施；\n"
        "4. 预警等级参考标准：流量 ≥ 5000m³/s 或水位超保证水位 → Ⅰ级；"
        "3000-5000m³/s 或超警戒水位 → Ⅱ级；2000-3000m³/s → Ⅲ级；其他 → Ⅳ级。"
    )
