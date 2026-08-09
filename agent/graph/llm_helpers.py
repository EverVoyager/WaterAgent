"""LLM 调用辅助。

从 workflow.py 拆分而来。依赖 errors（_classify_llm_error / LLMError）。
LangFuse 追踪集成：所有 LLM 调用自动记录到 LangFuse（未配置时为 no-op）。
"""
import json
import logging
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    APIError,
    RateLimitError,
)

from agent.graph.errors import LLMError, _classify_llm_error
from agent.tracing import trace_llm_call
from app.core.llm import LLM_TIMEOUTS, get_llm_client, get_llm_config, strip_think

logger = logging.getLogger(__name__)


def _call_llm_json(system: str, user: str, timeout_key: str = "default") -> Any:
    """调用 LLM 并解析为 JSON。LLM 失败或返回非 JSON 时抛 LLMError。"""
    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS[timeout_key])

    # P4.2 LangFuse 追踪
    with trace_llm_call(
        name=f"llm_json_{timeout_key}",
        model=settings["model"],
        input={"system": system[:500], "user": user[:500]},
        metadata={"timeout_key": timeout_key},
    ) as gen:
        try:
            resp = client.chat.completions.create(
                model=settings["model"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
        except (APITimeoutError, RateLimitError, APIConnectionError, APIError) as e:
            logger.exception("[_call_llm_json] LLM 调用失败 (%s)", type(e).__name__)
            raise _classify_llm_error(e) from e
        except Exception as e:
            logger.exception("[_call_llm_json] LLM 未知异常")
            raise _classify_llm_error(e) from e

        # 记录到 LangFuse
        content = strip_think((resp.choices[0].message.content or "").strip())
        gen.set_output({"content": content[:1000]})
        gen.set_usage(getattr(resp, "usage", None))

    # 兼容带 ```json ``` 包裹的输出
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
        if content.endswith("```"):
            content = content[:-3].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("[_call_llm_json] LLM 返回非 JSON: %s", content[:200])
        raise LLMError("format_error", f"LLM 返回格式异常（非 JSON）：{e}", status_code=502) from e


def _stream_llm(system: str, user: str, temperature: float = 0.7, max_tokens: int = 512,
                timeout_key: str = "chat"):
    """流式调用 LLM，逐 token yield。LLM 调用失败时抛 LLMError。"""
    settings = get_llm_config()
    client = get_llm_client().with_options(timeout=LLM_TIMEOUTS[timeout_key])

    # P4.2 LangFuse 追踪：流式调用累积输出后一次性记录
    with trace_llm_call(
        name=f"llm_stream_{timeout_key}",
        model=settings["model"],
        input={"system": system[:500], "user": user[:500]},
        metadata={"timeout_key": timeout_key, "stream": True},
    ) as gen:
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

        # 累积流式输出，结束后记录到 LangFuse
        # Qwen3 思考内容剥离：流式需缓冲检测 <think>...</think> 块
        collected: list[str] = []
        buffer = ""
        in_think = False
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if not content:
                continue
            collected.append(content)
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

        # 流结束后设置 output（截断到 1000 字符避免 trace 过大）
        full_output = "".join(collected)
        gen.set_output({"content": full_output[:1000]})
