"""结构化输出协议：工具通道、端点能力探测与降级。"""
import json
import logging
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)

from agent.graph.errors import LLMError, _classify_llm_error
from agent.prompts.synthesizer import SYNTH_RESPONSE_SCHEMA as _SYNTH_RESPONSE_SCHEMA
from app.core.llm_stats import record_llm_usage

logger = logging.getLogger(__name__)

# 结构化输出降级记忆（进程级）：第一档为 tool calling 强制 schema（流派一，
# DeepSeek 的 FC 支持良好）；端点若不支持 forced tool_choice（400）则降级到
# json_object。进程内记住探测结果，后续调用直接从上次结论开始，
# 避免每次都先吃一个 400 无效往返。
# None = 未探测；True = tool 通道可用；False = 不可用，从 json_object 开始
_SCHEMA_FORMAT_SUPPORTED: bool | None = None


def _reset_response_format_memory() -> None:
    """清空降级记忆（测试用）。"""
    global _SCHEMA_FORMAT_SUPPORTED, _TOOL_CHOICE_MODE
    _SCHEMA_FORMAT_SUPPORTED = None
    _TOOL_CHOICE_MODE = "required"


def _tool_def_from_schema(fmt: dict) -> list[dict]:
    """把 response_format 风格的 json_schema 定义转为 function tool 定义。

    流派一（对齐 Pydantic AI Tool Output / LangChain function_calling 方法）：
    结构化结果经工具调用通道提交，schema 作为工具参数由服务端参与约束，
    绕开 DeepSeek 不支持 response_format=json_schema 的限制。
    """
    inner = (fmt or {}).get("json_schema", {}) or {}
    return [{
        "type": "function",
        "function": {
            "name": inner.get("name", "submit_result"),
            "description": "提交结构化研判结果（系统会校验字段与规则一致性，"
                           "校验失败会通过 tool 结果返回修正建议）",
            "parameters": inner.get("schema", {"type": "object", "properties": {}}),
        },
    }]


def _extract_structured_content(msg) -> str:
    """从模型响应提取结构化 JSON 文本：工具通道优先，正文兜底。

    工具通道：tool_calls[0].function.arguments（模型按 schema 生成的参数）；
    兜底：message.content（部分端点忽略 forced tool_choice 时走正文 JSON）。
    MagicMock 响应（测试）的 tool_calls 非 list，自然落到 content 分支。
    """
    tcs = getattr(msg, "tool_calls", None)
    if isinstance(tcs, list):
        for tc in tcs:
            fn = getattr(tc, "function", None)
            args = getattr(fn, "arguments", None) if fn is not None else None
            if isinstance(args, str) and args.strip():
                return args.strip()
    return (getattr(msg, "content", None) or "").strip()


def _append_structured_retry_feedback(messages: list, msg, feedback: str) -> None:
    """校验失败重试：按模型响应形态回传反馈。

    工具通道（流派一）：追加 assistant(tool_calls + reasoning_content 原样回传)
    与 tool(validation_error) 配对消息——维持 tool_call/tool 配对约束
    （DeepSeek 思考模式续轮要求 reasoning_content 在场）。
    文本通道：assistant 正文 + user 反馈（原行为，向后兼容）。
    """
    tcs = getattr(msg, "tool_calls", None)
    if isinstance(tcs, list) and tcs:
        a_msg: dict[str, Any] = {
            "role": "assistant",
            "content": getattr(msg, "content", None) or None,
        }
        rc = getattr(msg, "reasoning_content", None)
        if rc:
            a_msg["reasoning_content"] = rc
        a_msg["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name,
                          "arguments": tc.function.arguments or "{}"}}
            for tc in tcs
        ]
        messages.append(a_msg)
        for tc in tcs:
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps({"validation_error": feedback}, ensure_ascii=False),
            })
    else:
        messages.append({"role": "assistant",
                         "content": (getattr(msg, "content", "") or "")[:2000]})
        messages.append({"role": "user", "content": feedback})

# 工具通道的 tool_choice 模式（进程级记忆）：
# "required" = 强制（OpenAI/vLLM 等支持）；DeepSeek 思考模式不支持 required/
# 命名强制 → 自动降级 "auto" 并在消息末尾附提交指令（单工具+明确指令≈强制）
_TOOL_CHOICE_MODE: str = "required"


def _call_tool_channel(client, model: str, messages: list, fmt: dict):
    """流派一工具通道：required → auto 自适应。

    DeepSeek 思考模式对 forced tool_choice 返回 400（"Thinking mode does
    not support this tool_choice"），此处自动改用 auto 并在调用副本末尾
    追加提交指令（不污染调用方 messages）；适配结果进程内记忆。
    """
    global _TOOL_CHOICE_MODE
    tool = _tool_def_from_schema(fmt)
    name = tool[0]["function"]["name"]
    send = list(messages)
    if _TOOL_CHOICE_MODE == "auto":
        send.append({"role": "user",
                     "content": f"请调用 {name} 工具提交本次结构化结果。"})
    try:
        return client.chat.completions.create(
            model=model, messages=send, temperature=0.3, max_tokens=8192,
            tools=tool, tool_choice=_TOOL_CHOICE_MODE,
        )
    except APIError as e:
        if (getattr(e, "status_code", None) == 400
                and _TOOL_CHOICE_MODE == "required" and "tool_choice" in str(e)):
            _TOOL_CHOICE_MODE = "auto"
            logger.warning(
                "[synthesizer] 端点思考模式不支持 forced tool_choice，"
                "本进程改用 auto + 提交指令"
            )
            return _call_tool_channel(client, model, messages, fmt)
        raise


def _call_synth_with_fallback(client, model: str, messages: list, schema=None):
    """分级降级调用 LLM：json_schema strict → json_object → 无 response_format。

    DeepSeek 等端点不支持 json_schema strict（400），报错则逐级降级；
    进程内记住探测结果（_SCHEMA_FORMAT_SUPPORTED），后续调用直接从上次
    结论开始，避免每次都先吃一个 400 无效往返。
    LLM 调用级异常（timeout/rate_limit/connection）直接抛 LLMError。

    Args:
        schema: 自定义 JSON schema（如 _SYNTH_META_SCHEMA）。默认 _SYNTH_RESPONSE_SCHEMA。

    注意：max_tokens 设为 8192，推理模型（如 deepseek-r1/v4）的 <think> 块
        会消耗大量 token，1500 不够会导致 JSON 输出被截断。
    """
    global _SCHEMA_FORMAT_SUPPORTED
    primary_schema = schema or _SYNTH_RESPONSE_SCHEMA
    # 尝试列表：从最强约束到最弱。
    # 第一档 tool_calling（流派一）：forced tool_choice，schema 走 tools 参数，
    # 服务端参与约束——兼容不支持 response_format=json_schema 的端点（DeepSeek）。
    # 已探测过 tool 通道不可用时跳过，直接从 json_object 开始
    all_formats = [
        ("tool_calling", primary_schema),
        ("json_object", {"type": "json_object"}),
        ("none", None),
    ]
    formats = all_formats[1:] if _SCHEMA_FORMAT_SUPPORTED is False else all_formats
    last_exc = None
    for label, fmt in formats:
        try:
            if label == "tool_calling":
                # required → auto 自适应（DeepSeek 思考模式适配）在通道内处理
                resp = _call_tool_channel(client, model, messages, fmt)
            else:
                kwargs = {"model": model, "messages": messages,
                          "temperature": 0.3, "max_tokens": 8192}
                if fmt is not None:
                    kwargs["response_format"] = fmt
                resp = client.chat.completions.create(**kwargs)
            record_llm_usage("synthesizer", resp.usage)
            if label == "tool_calling":
                _SCHEMA_FORMAT_SUPPORTED = True
            logger.info("[synthesizer] LLM 调用成功（结构化通道=%s）", label)
            return resp
        except (APITimeoutError, RateLimitError, APIConnectionError) as e:
            # 这类异常不重试，直接抛
            raise _classify_llm_error(e) from e
        except APIError as e:
            # 400 BadRequest：tool 通道或 response_format 不支持，降级重试
            if getattr(e, "status_code", None) == 400 and label != "none":
                if label == "tool_calling":
                    _SCHEMA_FORMAT_SUPPORTED = False
                    logger.warning(
                        "[synthesizer] forced tool_choice 不被当前端点支持，"
                        "本进程后续直接从 json_object 开始"
                    )
                else:
                    logger.warning("[synthesizer] response_format=%s 不支持，降级重试：%s", label, str(e)[:120])
                last_exc = e
                continue
            # 其他 APIError 直接抛
            raise _classify_llm_error(e) from e
        except Exception as e:
            logger.exception("[synthesizer] LLM 未知异常")
            raise _classify_llm_error(e) from e
    # 所有格式都失败（理论上不会走到，none 兜底）
    raise _classify_llm_error(last_exc) if last_exc else LLMError("api_error", "LLM 调用失败")
