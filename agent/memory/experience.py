"""经验检索与注入（Hermes 范式的应用层）。

将 MySQL 中积累的记忆/技能检索出来，注入到对应节点的 prompt 中：
- planner_node：注入相关技能（成功工具调用模式）+ 工具失败教训
- synthesizer_node：注入用户偏好 + 领域知识

注入策略：
- 控制长度（避免 prompt 膨胀）：每次最多 3 条技能 + 3 条偏好
- 相关性排序：按 hit_count / use_count 倒序
- 增量命中计数：被注入的记忆自动 hit_count + 1
"""
import logging
from typing import List

from agent.memory.memory_store import (
    MemoryType,
    get_memory_store,
    is_memory_enabled,
)

logger = logging.getLogger(__name__)


def get_relevant_experiences(query: str, limit: int = 3) -> str:
    """检索与当前 query 相关的经验，格式化为可注入 prompt 的文本。

    包含：
    - 相关技能（成功的工具调用模式）
    - 工具失败教训（避免重复犯错）

    用于 planner_node 注入。返回空字符串表示无经验或未启用。
    """
    if not is_memory_enabled():
        return ""

    store = get_memory_store()
    parts: List[str] = []

    # 1. 相关技能（成功的工具调用模式）
    try:
        skills = store.get_relevant_skills(query, limit=limit)
        if skills:
            parts.append("【过往成功经验】")
            for i, s in enumerate(skills, 1):
                tool_calls = s.get("tool_calls", [])
                tool_names = [tc.get("name", "") for tc in tool_calls]
                rounds_used = s.get("rounds_used", 1)
                use_count = s.get("use_count", 1)
                parts.append(
                    f"  {i}. 类似查询「{s.get('query_pattern', '')[:40]}」曾用 "
                    f"[{', '.join(tool_names)}] 在 {rounds_used} 轮内解决（已复用 {use_count} 次）"
                )
    except Exception as e:
        logger.debug("[experience] 检索技能失败：%s", e)

    # 2. 工具失败教训
    try:
        failures = store.get_memories(memory_type=MemoryType.TOOL_FAILURE, limit=3)
        if failures:
            parts.append("【工具失败教训（避免重复）】")
            for i, f in enumerate(failures, 1):
                content = f.get("content", "")[:80]
                parts.append(f"  {i}. {content}")
                # 命中计数 +1
                if f.get("id"):
                    store.increment_hit(f["id"])
    except Exception as e:
        logger.debug("[experience] 检索失败教训失败：%s", e)

    return "\n".join(parts) if parts else ""


def get_user_preferences() -> str:
    """检索用户偏好和领域知识，格式化为可注入 prompt 的文本。

    用于 synthesizer_node 注入。返回空字符串表示无偏好或未启用。
    """
    if not is_memory_enabled():
        return ""

    store = get_memory_store()
    parts: List[str] = []

    # 1. 用户偏好
    try:
        prefs = store.get_memories(memory_type=MemoryType.USER_PREFERENCE, limit=3)
        if prefs:
            parts.append("【用户偏好】")
            for i, p in enumerate(prefs, 1):
                content = p.get("content", "")[:80]
                parts.append(f"  {i}. {content}")
                if p.get("id"):
                    store.increment_hit(p["id"])
    except Exception as e:
        logger.debug("[experience] 检索偏好失败：%s", e)

    # 2. 用户纠正（曾被纠正过的错误，避免再犯）
    try:
        corrections = store.get_memories(memory_type=MemoryType.USER_CORRECTION, limit=2)
        if corrections:
            parts.append("【历史纠正（避免再犯）】")
            for i, c in enumerate(corrections, 1):
                content = c.get("content", "")[:80]
                parts.append(f"  {i}. {content}")
                if c.get("id"):
                    store.increment_hit(c["id"])
    except Exception as e:
        logger.debug("[experience] 检索纠正失败：%s", e)

    # 3. 领域知识
    try:
        knowledge = store.get_memories(memory_type=MemoryType.DOMAIN_KNOWLEDGE, limit=5)
        if knowledge:
            parts.append("【已积累领域知识】")
            for i, k in enumerate(knowledge, 1):
                content = k.get("content", "")[:100]
                parts.append(f"  {i}. {content}")
                if k.get("id"):
                    store.increment_hit(k["id"])
    except Exception as e:
        logger.debug("[experience] 检索领域知识失败：%s", e)

    return "\n".join(parts) if parts else ""
