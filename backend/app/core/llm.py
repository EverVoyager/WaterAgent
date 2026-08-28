"""LLM 客户端封装：基于 OpenAI 兼容接口。"""
import re
from functools import lru_cache

import httpx
from openai import OpenAI
from qdrant_client import QdrantClient

from agent.utils import WARNING_THRESHOLDS
from app.core.config import get_settings

# Qwen3 思考内容剥离：本地微调模型在 nothink 模板下仍可能输出 <think>...</think>
# 后端统一剥离，避免思考内容泄漏到前端 / 干扰下游节点解析
_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """剥离 Qwen3 <think>...</think> 思考内容。

    本地微调的 Qwen3-4B 即使配置 qwen3_nothink 模板，推理时仍可能输出
    思考块。LlamaFactory API 不会自动剥离，需在客户端层统一处理。
    """
    if not text:
        return text
    return _THINK_PATTERN.sub("", text).lstrip("\n")


def extract_content(msg) -> str:
    """从 LLM 响应的 message 对象中提取文本内容。

    仅取 message.content，不回退到 reasoning_content。

    主流方案（OpenAI / LangChain / DeepSeek / browser-use）均将推理过程与答案
    物理分离：reasoning_content 是推理模型的内部思考链，可能包含系统提示词、
    中间推理等不应暴露给用户的内容。将其回退为 content 会导致思考链泄漏到
    前端（曾导致 direct_chat 把完整推理过程当作答案返回的 bug）。

    content 为空时返回空字符串，由调用方决定是否抛错。
    """
    content = (getattr(msg, "content", None) or "").strip()
    return strip_think(content)


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
    f1 = WARNING_THRESHOLDS["flow_level1"]
    f2 = WARNING_THRESHOLDS["flow_level2"]
    f3 = WARNING_THRESHOLDS["flow_level3"]
    return (
        "你是黄河吕梁段防汛预警智能体。你的职责是根据用户问题，"
        "调用合适的工具获取天气、水情、径流预测、GIS 地形、法规政策等信息，"
        "并基于工具返回的数据进行综合研判，输出预警等级与具体应急预案。\n\n"
        "规则：\n"
        "1. 一次可以调用多个工具，但每次只调用最相关的 1-3 个；\n"
        "2. 工具调用失败时，应说明原因并基于已有信息进行研判；\n"
        "3. 最终回答必须包含：预警等级（Ⅰ/Ⅱ/Ⅲ/Ⅳ）、研判依据、具体应急措施；\n"
        f"4. 预警等级参考标准：流量 ≥ {f1}m³/s 或水位超保证水位 → Ⅰ级；"
        f"{f2}-{f1}m³/s 或超警戒水位 → Ⅱ级；{f3}-{f2}m³/s → Ⅲ级；其他 → Ⅳ级。"
    )
