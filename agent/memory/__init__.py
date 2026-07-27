"""Hermes 范式自进化记忆模块。

三层记忆体系：
- 短期记忆：本次对话历史 + 工具调用轨迹（内存中，会话结束清除）
- 长期记忆：跨会话的用户偏好、纠正、领域知识（MySQL 持久化）
- 技能记忆：成功的工具调用模式（query 模式 → 工具组合）（MySQL 持久化）

反思循环（reflection.py）：
- 触发条件：用户纠正、工具失败、格式错误、多轮解决
- 异步执行：不阻塞响应
- 输出：提取经验写入长期/技能记忆

经验注入（experience.py）：
- 在 planner_node 注入"过往经验"few-shot
- 在 synthesizer_node 注入"用户偏好"
"""
from agent.memory.experience import (
    get_relevant_experiences,
    get_user_preferences,
)
from agent.memory.memory_store import (
    MemoryStore,
    MemoryType,
    get_memory_store,
    is_memory_enabled,
)
from agent.memory.reflection import (
    should_reflect,
    run_reflection_async,
)

__all__ = [
    # memory_store
    "MemoryStore",
    "MemoryType",
    "get_memory_store",
    "is_memory_enabled",
    # reflection
    "should_reflect",
    "run_reflection_async",
    # experience
    "get_relevant_experiences",
    "get_user_preferences",
]
