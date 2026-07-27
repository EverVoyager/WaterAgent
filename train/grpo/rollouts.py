"""GRPO 工具回放：与数据生成共用同一确定性回放逻辑。

说明：trl GRPOTrainer 标准流程为单轮补全——模型在一次生成中输出完整
调用计划（多个 <tool_call> 块）+ 最终研判段，无需交互式多轮 rollout；
本模块只提供确定性回放（供多轮扩展与评估复用）。
"""
from agent.tools import mock_executor
from train.data_gen.scenario import Scenario


def replay_tool_call(scn: Scenario, name: str, arguments: dict) -> dict:
    overrides = scn.tool_overrides.get(name)
    seed = abs(hash(f"{scn.scenario_id}:{name}")) % (2**31)
    return mock_executor.execute_tool(name, arguments, overrides=overrides, seed=seed)
