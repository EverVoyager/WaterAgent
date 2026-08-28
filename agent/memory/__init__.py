"""五类记忆架构（对齐认知科学分类 + Claude Code / Codex 双层记忆模式）。

- 会话记忆：chat_sessions/chat_messages + context_compact（当前会话上下文）
- 长期记忆：MEMORY.md（用户权威手册）+ memory/ 目录（Agent 自动记忆）——longterm.py
- 语义记忆：agent_semantic 表 + 向量索引（领域知识/文档）——semantic_store.py
- 情景记忆：agent_episodes 表 + 向量索引（事件与解决方法）——episode_store.py
- 程序记忆：agent_procedures 表 + 向量索引（可晋升 Skill 的通用方法）——procedure_store.py
- 反思审计：agent_reflections 表——memory_store.py

反思循环（reflection.py）：触发条件不变，输出分发到长期/语义/情景/程序四类。
经验注入（experience.py）：planner 注入情景+程序，synthesizer 注入语义，
长期记忆经 longterm.build_longterm_section 常驻三处 system prompt。
"""
from agent.memory.experience import (
    clear_injected_tracking,
    finalize_injected_tracking,
    get_injected_memories,
    get_relevant_experiences,
    get_semantic_knowledge,
)
from agent.memory.longterm import (
    apply_longterm_edits,
    build_longterm_section,
    load_longterm_memory,
)
from agent.memory.memory_store import (
    MemoryStore,
    get_memory_store,
    is_memory_enabled,
)
from agent.memory.reflection import (
    run_reflection_async,
    should_reflect,
)

__all__ = [
    # memory_store（反思审计）
    "MemoryStore",
    "get_memory_store",
    "is_memory_enabled",
    # reflection
    "should_reflect",
    "run_reflection_async",
    # experience（注入聚合）
    "get_relevant_experiences",
    "get_semantic_knowledge",
    "clear_injected_tracking",
    "get_injected_memories",
    "finalize_injected_tracking",
    # longterm（双层文件）
    "load_longterm_memory",
    "build_longterm_section",
    "apply_longterm_edits",
]
