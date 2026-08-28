"""SSE 流式接口单元测试（离线，mock Agent 图入口）。

覆盖 _stream_generator 的桥接逻辑：
- 正常事件按序透传（reasoning_step/tool_call/tool_result/synth_meta/answer_delta/done）
- 队列空闲时发心跳 comment 保活
- LLMError → error 事件携带 kind/status_code；未知异常不带 kind
- 请求体校验失败返回 422
"""
import json
import time

import pytest
from fastapi.testclient import TestClient

from agent.graph.errors import LLMError
from app.main import create_app


@pytest.fixture()
def client():
    return TestClient(create_app())


def _parse_sse(raw: str) -> tuple[list[dict], int]:
    """把 SSE 响应原文解析为 (JSON 事件列表, 心跳 comment 数)。"""
    events: list[dict] = []
    keepalives = 0
    for line in raw.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
        elif line.startswith(":"):
            keepalives += 1
    return events, keepalives


def _sample_events():
    yield {"type": "reasoning_step", "step": "planner", "phase": "start",
           "message": "规划中", "details": {}}
    yield {"type": "tool_call", "tool": "get_hydrology",
           "arguments": {"station": "吴堡"}, "round": 1}
    yield {"type": "tool_result", "tool": "get_hydrology",
           "result": {"station": "吴堡"}, "error": "", "round": 1}
    yield {"type": "synth_meta",
           "data": {"warning_level": "II", "reasoning": "流量超警戒",
                    "actions": ["启动Ⅱ级响应"], "citations": []}}
    yield {"type": "answer_delta", "content": "当前"}
    yield {"type": "answer_delta", "content": "水情超警戒"}
    yield {"type": "done",
           "data": {"answer": "当前水情超警戒", "warning_level": "II",
                    "reasoning": "流量超警戒", "actions": ["启动Ⅱ级响应"],
                    "citations": [], "tool_calls": [], "rounds": 1,
                    "intent": "agent_task"}}


def _stream_once(client: TestClient, query: str = "吴堡站现在水情怎么样") -> str:
    with client.stream(
        "POST", "/api/agent/query/stream", json={"query": query},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers.get("cache-control") == "no-cache"
        return "".join(resp.iter_text())


def test_stream_events_passthrough_in_order(client, monkeypatch):
    """全部事件按产出顺序透传，done 为最后一条且数据完整。"""
    monkeypatch.setattr(
        "app.api.agent.run_graph_agent_stream_v2", lambda **kw: _sample_events(),
    )
    raw = _stream_once(client)
    events, _ = _parse_sse(raw)
    types = [e["type"] for e in events]
    assert types == [
        "reasoning_step", "tool_call", "tool_result",
        "synth_meta", "answer_delta", "answer_delta", "done",
    ]
    # synth_meta 先于 answer_delta 到达（前端先渲染预警卡再流式补答案）
    assert types.index("synth_meta") < types.index("answer_delta")
    done_data = events[-1]["data"]
    assert done_data["answer"] == "当前水情超警戒"
    assert done_data["warning_level"] == "II"


def test_stream_keepalive_comment(client, monkeypatch):
    """生成器迟迟不出数时主线程发心跳 comment。"""
    monkeypatch.setattr("app.api.agent.SSE_KEEPALIVE_INTERVAL", 0.05)

    def _slow_events():
        time.sleep(0.3)
        yield from _sample_events()

    monkeypatch.setattr(
        "app.api.agent.run_graph_agent_stream_v2", lambda **kw: _slow_events(),
    )
    raw = _stream_once(client)
    _, keepalives = _parse_sse(raw)
    assert keepalives >= 1
    # 心跳不打断后续事件完整性
    events, _ = _parse_sse(raw)
    assert events[-1]["type"] == "done"


def test_stream_llm_error_carries_kind_and_status(client, monkeypatch):
    """LLMError 映射为 error 事件并保留 kind/status_code（非流式接口同款分类）。"""

    def _raise_before_stream(**kw):
        raise LLMError("timeout", "LLM 调用超时：60s", status_code=504)
        yield  # pragma: no cover

    monkeypatch.setattr(
        "app.api.agent.run_graph_agent_stream_v2", _raise_before_stream,
    )
    raw = _stream_once(client)
    events, _ = _parse_sse(raw)
    errs = [e for e in events if e["type"] == "error"]
    assert len(errs) == 1
    assert errs[0]["kind"] == "timeout"
    assert errs[0]["status_code"] == 504
    assert "LLM 调用超时" in errs[0]["message"]
    # 无 done 事件
    assert all(e["type"] != "done" for e in events)


def test_stream_unknown_error_has_no_kind(client, monkeypatch):
    """未知异常（如工具 NameError 回归）也封装为 error 事件，不带 kind 字段。"""

    def _boom(**kw):
        yield {"type": "answer_delta", "content": "部分输出"}
        raise RuntimeError("unexpected boom")

    monkeypatch.setattr(
        "app.api.agent.run_graph_agent_stream_v2", lambda **kw: _boom(),
    )
    raw = _stream_once(client)
    events, _ = _parse_sse(raw)
    errs = [e for e in events if e["type"] == "error"]
    assert len(errs) == 1
    assert "kind" not in errs[0]
    assert "unexpected boom" in errs[0]["message"]
    # error 前已推送的 answer_delta 不回滚（已产出的部分保留）
    assert any(e["type"] == "answer_delta" for e in events)


def test_stream_invalid_body_422(client):
    """query 为空串校验失败返回 422，不进入流式响应。"""
    resp = client.post("/api/agent/query/stream", json={"query": ""})
    assert resp.status_code == 422
