"""教师模型合成 Hermes 轨迹（DashScope qwen-plus）。

- 双轨消息：发给 API 的 api_messages 用原生 tool_calls/tool_call_id 结构
  （OpenAI 兼容服务硬性要求）；落盘的 messages 用 Hermes 文本格式（训练格式）
- 工具结果确定性回放：execute_tool(overrides=场景覆盖值, seed=hash(scenario_id))
- 强制走 mock：回放前屏蔽 real_executor，避免依赖 Qdrant/外部 API
- 限速：简单令牌间隔（60/rpm 秒）；断点续传：按 scenario_id 跳过
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

from agent.tools import mock_executor
from agent.tools.schemas import build_openai_tools
from app.core.llm import get_default_system_prompt
from train.data_gen.hermes_format import make_tool_call_text, make_tool_response_text
from train.data_gen.scenario import Scenario

logger = logging.getLogger(__name__)


def _force_mock() -> None:
    """让 execute_tool 的真实实现分支永远不可用（仅作用于本进程）。"""
    import agent.tools.real_executor as re_mod

    def _unavailable(*a, **k):
        raise RuntimeError("data-gen: real executor disabled")

    re_mod.real_execute_tool = _unavailable


def _replay_tool(scn: Scenario, name: str, arguments: dict) -> dict:
    overrides = scn.tool_overrides.get(name)
    seed = abs(hash(f"{scn.scenario_id}:{name}")) % (2**31)
    return mock_executor.execute_tool(name, arguments, overrides=overrides, seed=seed)


def synthesize_one(client, model: str, scn: Scenario, max_rounds: int = 8) -> Optional[list]:
    """单场景多轮合成。达到轮次上限或教师输出非法 → None。

    返回 Hermes 文本格式轨迹（落盘用），不含 API 原生 tool_calls 结构。
    """
    _force_mock()
    # 落盘轨迹（Hermes 文本格式）
    messages = [
        {"role": "system", "content": get_default_system_prompt()},
        {"role": "user", "content": scn.query},
    ]
    # API 消息（原生格式；assistant 带 tool_calls 数组、tool 带 tool_call_id）
    api_messages = list(messages)
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=model, messages=api_messages, tools=build_openai_tools(), temperature=0.7,
        )
        msg = resp.choices[0].message
        sdk_calls = getattr(msg, "tool_calls", None) or []
        if not sdk_calls:
            if not msg.content:
                return None
            messages.append({"role": "assistant", "content": msg.content})
            return messages
        # Hermes 文本：拼接本轮全部调用块
        hermes_text = ""
        for tc in sdk_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                return None
            hermes_text += make_tool_call_text(tc.function.name, args)
        messages.append({"role": "assistant", "content": hermes_text})
        # API 侧：原生 assistant tool_calls 消息 + 逐个 tool 回复
        api_messages.append(msg)
        for tc in sdk_calls:
            args = json.loads(tc.function.arguments)
            try:
                result = _replay_tool(scn, tc.function.name, args)
            except ValueError:
                return None  # 教师产出非法工具/参数，直接丢弃（不进过滤流程）
            messages.append({"role": "tool", "content": make_tool_response_text(result)})
            api_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
    return None


def load_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["scenario_id"])
    return done


def synthesize_dataset(client, model: str, scenarios: list, out_path: Path, rpm: int = 30) -> int:
    """批量合成 + 追加写盘 + 断点续传。返回本次新写入条数。"""
    done = load_done_ids(out_path)
    interval = 60.0 / max(rpm, 1)
    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for scn in scenarios:
            if scn.scenario_id in done:
                continue
            t0 = time.time()
            try:
                trace = synthesize_one(client, model, scn)
            except Exception as e:
                logger.warning("[teacher] %s 合成异常跳过: %s", scn.scenario_id, e)
                trace = None
            if trace is not None:
                f.write(json.dumps({
                    "scenario_id": scn.scenario_id,
                    "level": scn.expected_level,
                    "query_type": scn.query_type,
                    "messages": trace,
                }, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
            time.sleep(max(0.0, interval - (time.time() - t0)))
    return written
