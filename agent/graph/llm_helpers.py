"""LLM 调用辅助。

从 workflow.py 拆分而来。依赖 errors（_classify_llm_error / LLMError）。
"""
import logging
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)

from agent.graph.errors import LLMError, _classify_llm_error
from agent.utils import parse_json_from_llm
from app.core.llm import LLM_TIMEOUTS, extract_content, get_llm_client, get_llm_config

logger = logging.getLogger(__name__)


def _call_llm_json(system: str, user: str, timeout_key: str = "default") -> Any:
    """调用 LLM 并解析为 JSON。LLM 失败或返回非 JSON 时抛 LLMError。"""
    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS[timeout_key])
    try:
        resp = client.chat.completions.create(
            model=settings["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
    except (APITimeoutError, RateLimitError, APIConnectionError, APIError) as e:
        logger.exception("[_call_llm_json] LLM 调用失败 (%s)", type(e).__name__)
        raise _classify_llm_error(e) from e
    except Exception as e:
        logger.exception("[_call_llm_json] LLM 未知异常")
        raise _classify_llm_error(e) from e

    # extract_content 仅取 message.content（不回退 reasoning_content）
    content = extract_content(resp.choices[0].message)

    # 解析 JSON（多策略容错：去代码块包裹、大括号配对、修复单引号/尾随逗号）
    result = parse_json_from_llm(content)
    if result is None:
        logger.error("[_call_llm_json] LLM 返回非 JSON: %s", content[:200])
        raise LLMError("format_error", "LLM 返回格式异常（非 JSON）", status_code=502)
    return result


def _stream_llm(system: str, user: str, temperature: float = 0.7, max_tokens: int = 2048,
                timeout_key: str = "chat"):
    """流式调用 LLM，逐 token yield。LLM 调用失败时抛 LLMError。"""
    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS[timeout_key])
    try:
        stream = client.chat.completions.create(
            model=settings["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
    except (APITimeoutError, RateLimitError, APIConnectionError, APIError) as e:
        logger.exception("[_stream_llm] LLM 流式调用失败 (%s)", type(e).__name__)
        raise _classify_llm_error(e) from e
    except Exception as e:
        logger.exception("[_stream_llm] LLM 未知异常")
        raise _classify_llm_error(e) from e

    # Qwen3 思考内容剥离：流式需缓冲检测 <think>...</think> 块
    # 仅取 delta.content，忽略 delta.reasoning_content（推理过程不流式推给前端）
    buffer = ""
    in_think = False
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if not content:
            continue
        buffer += content
        # 状态机剥离 <think>...</think>
        output = ""
        while buffer:
            if in_think:
                end = buffer.find("</think>")
                if end == -1:
                    # think 块未结束，继续缓冲
                    break
                # 找到结束标签，跳过 think 内容
                buffer = buffer[end + len("</think>"):]
                # 跳过后缀空白
                while buffer and buffer[0] in " \n\t":
                    buffer = buffer[1:]
                in_think = False
            else:
                start = buffer.find("<think>")
                if start == -1:
                    # 没有 think 标签，但要保留可能的不完整 "<think"
                    # 检查 buffer 末尾是否是 "<think" 的前缀
                    partial = 0
                    for plen in range(min(len(buffer), len("<think>")), 0, -1):
                        if buffer.endswith("<think>"[:plen]):
                            partial = plen
                            break
                    output += buffer[:len(buffer) - partial]
                    buffer = buffer[len(buffer) - partial:]
                    break
                # 找到 think 开始标签
                output += buffer[:start]
                buffer = buffer[start + len("<think>"):]
                in_think = True
            if output:
                yield output
