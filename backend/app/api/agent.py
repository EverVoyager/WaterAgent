"""Agent 对话接口。"""
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.graph.workflow import LLMError, run_graph_agent, run_graph_agent_stream_v2

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])

# SSE 心跳间隔（秒）。长连接无数据时定期发送 comment 保活，
# 避免代理/负载均衡提前断开（参考 nginx proxy_read_timeout 默认 60s）
SSE_KEEPALIVE_INTERVAL = 15.0


# ====== 请求 / 响应模型 ======

class ChatMessage(BaseModel):
    """对话消息。"""

    role: str = Field(..., description="user/assistant")
    content: str = Field(..., description="消息内容")


class AgentQueryRequest(BaseModel):
    """Agent 查询请求。"""

    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    system_prompt: Optional[str] = Field(None, description="自定义系统提示词（暂未使用，预留）")
    history: List[ChatMessage] = Field(
        default_factory=list, description="历史对话（不含当前问题）"
    )


class ToolCallInfo(BaseModel):
    """工具调用信息。"""

    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    error: str = ""
    round: int = 1


class AgentQueryResponse(BaseModel):
    """Agent 查询响应（LangGraph 版本）。"""

    answer: str = Field(..., description="最终自然语言回答")
    warning_level: str = Field("", description="预警等级 I/II/III/IV；闲聊为空")
    reasoning: str = Field("", description="研判依据；闲聊为空")
    actions: List[str] = Field(default_factory=list, description="应急措施列表")
    tool_calls: List[ToolCallInfo] = Field(
        default_factory=list, description="工具调用链路"
    )
    rounds: int = Field(0, description="实际执行的轮次")
    intent: str = Field("agent_task", description="意图：chitchat / agent_task")


# ====== 接口 ======

@router.post("/query", response_model=AgentQueryResponse)
def agent_query(req: AgentQueryRequest) -> AgentQueryResponse:
    """接收用户问题，运行 LangGraph Agent，返回预警研判结果。"""
    logger.info("Agent query (graph): %s", req.query[:200])

    try:
        history = [m.model_dump() for m in req.history]
        result = run_graph_agent(user_query=req.query, history=history)
    except LLMError as e:
        # P2：LLM 异常按分类返回对应 HTTP 状态码
        logger.warning("[agent_query] LLM error: %s (kind=%s)", e, e.kind)
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
    except Exception as e:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=f"Agent 运行失败: {e}") from e

    return AgentQueryResponse(
        answer=result["final_answer"],
        warning_level=result["warning_level"],
        reasoning=result["reasoning"],
        actions=result["actions"],
        tool_calls=[
            ToolCallInfo(
                tool_name=tc["tool_name"],
                arguments=tc["arguments"],
                result=tc["result"],
                error=tc.get("error", ""),
                round=tc.get("round", 1),
            )
            for tc in result["tool_calls"]
        ],
        rounds=result["rounds"],
        intent=result.get("intent", "agent_task"),
    )


# ====== SSE 流式接口 ======

def _sse_event(data: dict) -> str:
    """格式化为 SSE 事件字符串。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_keepalive_comment() -> str:
    """SSE comment 心跳（客户端忽略，仅用于保活）。"""
    return ": keep-alive\n\n"


def _stream_generator(query: str, history: list):
    """SSE 事件生成器（P5 加心跳保活）。

    使用独立线程跑 Agent，主线程定期发心跳避免连接超时。
    """
    # 用 queue 把 Agent 事件传到主线程
    import queue
    event_queue: "queue.Queue[Any]" = queue.Queue()
    # 哨兵对象标记 Agent 结束
    _SENTINEL = object()

    def _agent_worker():
        try:
            for event in run_graph_agent_stream_v2(user_query=query, history=history):
                event_queue.put(event)
        except LLMError as e:
            # P2：LLM 异常封装为 error 事件推给前端
            logger.warning("[sse] LLM error: %s (kind=%s)", e, e.kind)
            event_queue.put({"type": "error", "message": str(e),
                             "kind": e.kind, "status_code": e.status_code})
        except Exception as e:
            logger.exception("[sse] 流式生成失败")
            event_queue.put({"type": "error", "message": str(e)})
        finally:
            event_queue.put(_SENTINEL)

    worker_thread = threading.Thread(target=_agent_worker, daemon=True)
    worker_thread.start()

    last_emit = time.time()
    try:
        while True:
            try:
                # 最多等 keepalive 间隔，超时就发心跳
                item = event_queue.get(timeout=max(0.1, SSE_KEEPALIVE_INTERVAL - (time.time() - last_emit)))
            except queue.Empty:
                # 发心跳
                yield _sse_keepalive_comment()
                last_emit = time.time()
                continue

            if item is _SENTINEL:
                break
            yield _sse_event(item)
            last_emit = time.time()
    finally:
        if worker_thread.is_alive():
            # 主线程退出时 worker 仍可能在跑，daemon=True 会随进程退出
            logger.warning("[sse] stream ended before agent finished, worker still running")


@router.post("/query/stream")
def agent_query_stream(req: AgentQueryRequest):
    """Agent 对话流式接口（SSE）。

    返回 text/event-stream，事件类型：
      - reasoning_step：推理步骤（router/planner/executor/reflector/synthesizer/direct_chat）
      - intent：意图识别结果
      - tool_call / tool_result：工具调用链
      - synth_meta：综合研判结构化结论（warning_level/reasoning/actions）
      - answer_delta：token 级流式答案
      - done：完整响应
      - error：错误

    每条事件格式：data: {"type": "...", ...}\n\n
    """
    logger.info("Agent query (stream v2): %s", req.query[:200])
    history = [m.model_dump() for m in req.history]
    return StreamingResponse(
        _stream_generator(req.query, history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )
