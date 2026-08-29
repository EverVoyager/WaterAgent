"""评估环境层：可重置的环境状态（书中"评估环境五要素"之二）。

mock_executor 的 overrides+seed 回放分支是现成的环境机制（训练侧 Task 3
引入），本模块把它包装成 case 粒度的环境上下文：

- with case_env(case): 进入确定性回放环境（该 case 的 overrides+seed 生效，
  所有工具强制走 mock，工具结果缓存绕过）
- 退出即恢复，环境状态可重置——任何 case 在任何时刻重跑结果一致

环境要求：真实实现（RAG/爬虫）返回非确定数据，会破坏可复现性，
回放激活期间被 mock_executor 强制旁路，无需在此额外处理。
"""
from contextlib import contextmanager
from typing import Any

from agent.tools.mock_executor import (
    clear_replay_context,
    is_replay_active,
    set_replay_context,
)


@contextmanager
def case_env(case):
    """进入单条用例的确定性回放环境。

    with case_env(case):
        result = run_graph_agent(case.query)

    进入时设置 overrides+seed，退出时无条件清理（异常路径同样恢复），
    保证评估环境可重置。嵌套使用以外层为准（评估按 case 顺序执行，不会嵌套）。
    """
    if is_replay_active():
        # 防御：上一个 case 忘记清理会污染本 case 环境
        clear_replay_context()
    set_replay_context(case.overrides, case.seed)
    try:
        yield
    finally:
        clear_replay_context()


def env_snapshot() -> dict[str, Any]:
    """当前环境状态快照（报告 config 段用，说明评估在什么环境下执行）。"""
    return {
        "replay_active": is_replay_active(),
    }
