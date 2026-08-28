"""反思审计存储测试（五类记忆架构版，memory_store 只剩 agent_reflections 职责）。

覆盖：
- 未启用（MYSQL_PASSWORD 空）时的行为
- MySQL 不可达时的失败传播
- 单例与全局开关
- 需 MySQL 的 CRUD 走 skipif（与 test_skills 同模式）
"""
import pytest

from agent.memory.memory_store import MemoryStore, get_memory_store, is_memory_enabled


def _disabled_store() -> MemoryStore:
    return MemoryStore(host="127.0.0.1", port=3306, user="root",
                       password="", database="water_agent")


class TestDisabledStore:
    """MYSQL_PASSWORD 为空 → enabled=False，操作抛 RuntimeError（调用方捕获降级）。"""

    def test_disabled_property(self):
        assert _disabled_store().enabled is False

    def test_add_reflection_raises(self):
        with pytest.raises(RuntimeError, match="未启用"):
            _disabled_store().add_reflection(
                user_query="q", trigger_reason="multi_round", reflection_text="r")

    def test_list_reflections_raises(self):
        with pytest.raises(RuntimeError, match="未启用"):
            _disabled_store().list_reflections()


class TestConnectionFailure:
    """MySQL 配置了但不可达 → 抛异常（BaseStore 事务型连接直接传播）。"""

    def test_add_reflection_connection_error(self):
        store = MemoryStore(host="127.0.0.1", port=13306, user="root",
                            password="x", database="nodb")
        assert store.enabled is True
        with pytest.raises(RuntimeError):
            store.add_reflection(
                user_query="q", trigger_reason="multi_round", reflection_text="r")


class TestSingletonAndGlobals:
    def test_get_memory_store_singleton(self):
        assert get_memory_store() is get_memory_store()

    def test_is_memory_enabled_reflects_password(self):
        # 全局单例语义：is_memory_enabled 与单例 enabled 一致
        assert is_memory_enabled() == get_memory_store().enabled


# ====== 需要 MySQL 的 CRUD（CI 由 mysql service 提供，本地未配置自动跳过）======

_requires_mysql = pytest.mark.skipif(
    not is_memory_enabled(),
    reason="MemoryStore 需要 MySQL（未配置 MYSQL_PASSWORD）",
)


@_requires_mysql
class TestReflectionAuditCRUD:
    def test_add_and_list_reflection(self):
        store = get_memory_store()
        rid = store.add_reflection(
            user_query="吴堡站水情怎么样",
            trigger_reason="multi_round",
            tool_calls_summary="get_hydrology ✓",
            final_answer="当前流量...",
            reflection_text="经验已沉淀",
            memories_created=2,
        )
        assert isinstance(rid, int)
        rows = store.list_reflections(limit=10)
        assert any(r["id"] == rid for r in rows)
        matched = next(r for r in rows if r["id"] == rid)
        assert matched["trigger_reason"] == "multi_round"
        assert matched["memories_created"] == 2
