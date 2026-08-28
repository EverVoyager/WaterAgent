"""MemoryStore 单元测试。

覆盖：
1. 密码为空时 enabled=False，所有方法返回空/None（降级行为）
2. MySQL 连接失败时的降级行为（mock pymysql.connect 抛异常）
3. get_memory_store 单例 + is_memory_enabled 逻辑
4. 记忆/技能/反思的写入与检索（mock 连接与游标）
"""
from unittest.mock import MagicMock, patch

from agent.memory.memory_store import (
    MemoryStore,
    MemoryType,
    get_memory_store,
    is_memory_enabled,
)

# ====== 降级：密码为空 ======

class TestDisabledStore:
    """MYSQL_PASSWORD 为空时，MemoryStore 应禁用并返回空结果。"""

    def setup_method(self):
        self.store = MemoryStore(
            host="localhost", port=3306,
            user="root", password="", database="test",
        )

    def test_disabled_property(self):
        assert self.store.enabled is False

    def test_add_memory_returns_none(self):
        result = self.store.add_memory(MemoryType.USER_PREFERENCE, "test")
        assert result is None

    def test_get_memories_returns_empty(self):
        assert self.store.get_memories() == []

    def test_increment_hit_noop(self):
        # 不应抛异常
        self.store.increment_hit(1)

    def test_add_skill_noop(self):
        self.store.add_skill("test query", [{"tool": "get_hydrology"}], True)

    def test_get_relevant_skills_returns_empty(self):
        assert self.store.get_relevant_skills("水情查询") == []

    def test_add_reflection_noop(self):
        self.store.add_reflection("query", "tool_failure", "reflection text")

    def test_compact_memories_returns_negative(self):
        assert self.store.compact_memories(MemoryType.USER_PREFERENCE, lambda *_: []) == -1


# ====== 降级：MySQL 连接失败 ======

class TestConnectionFailure:
    """enabled=True 但 MySQL 连接失败时，应降级返回空结果且不抛异常。"""

    def setup_method(self):
        self.store = MemoryStore(
            host="localhost", port=3306,
            user="root", password="secret", database="test",
        )

    def test_add_memory_connection_error(self):
        """连接失败时 add_memory 返回 None，不抛异常。"""
        with patch("pymysql.connect", side_effect=Exception("Connection refused")):
            result = self.store.add_memory(MemoryType.USER_PREFERENCE, "test")
        assert result is None

    def test_get_memories_connection_error(self):
        with patch("pymysql.connect", side_effect=Exception("Connection refused")):
            result = self.store.get_memories()
        assert result == []

    def test_get_relevant_skills_connection_error(self):
        with patch("pymysql.connect", side_effect=Exception("Connection refused")):
            result = self.store.get_relevant_skills("水情查询")
        assert result == []

    def test_add_skill_connection_error(self):
        with patch("pymysql.connect", side_effect=Exception("Connection refused")):
            self.store.add_skill("test", [{"tool": "x"}], True)
        # 不抛异常即通过

    def test_add_reflection_connection_error(self):
        with patch("pymysql.connect", side_effect=Exception("Connection refused")):
            self.store.add_reflection("q", "reason", "text")
        # 不抛异常即通过

    def test_increment_hit_connection_error(self):
        with patch("pymysql.connect", side_effect=Exception("Connection refused")):
            self.store.increment_hit(1)
        # 不抛异常即通过


# ====== 正常流程（mock 连接与游标）======

class TestNormalOperations:
    """mock MySQL 连接，验证 SQL 执行与结果解析。"""

    def setup_method(self):
        self.store = MemoryStore(
            host="localhost", port=3306,
            user="root", password="secret", database="test",
        )
        self.store._initialized = True  # 跳过建表

    def test_add_memory_success(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 42
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("pymysql.connect", return_value=mock_conn):
            result = self.store.add_memory(
                MemoryType.DOMAIN_KNOWLEDGE,
                "龙门站警戒水位 377.5m",
                context={"station": "龙门"},
                tags=["hydrology", "龙门"],
            )
        assert result == 42
        # 验证 SQL 参数
        args = mock_cursor.execute.call_args[0]
        assert "INSERT INTO agent_memories" in args[0]
        assert args[1][0] == "domain_knowledge"
        assert args[1][1] == "龙门站警戒水位 377.5m"

    def test_get_memories_with_filter(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"id": 1, "memory_type": "user_preference", "content": "不用 emoji",
             "context": None, "tags": None, "hit_count": 3, "created_at": "2024-01-01"},
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("pymysql.connect", return_value=mock_conn):
            result = self.store.get_memories(
                memory_type=MemoryType.USER_PREFERENCE,
                limit=10,
            )
        assert len(result) == 1
        assert result[0]["content"] == "不用 emoji"
        assert result[0]["hit_count"] == 3

    def test_get_relevant_skills_parses_json(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"query_pattern": "水情查询", "tool_calls_json": '[{"tool":"get_hydrology"}]',
             "success": True, "rounds_used": 1, "use_count": 5},
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("pymysql.connect", return_value=mock_conn):
            result = self.store.get_relevant_skills("龙门站水情", limit=3)
        assert len(result) == 1
        assert result[0]["tool_calls"] == [{"tool": "get_hydrology"}]
        assert result[0]["use_count"] == 5

    def test_get_relevant_skills_invalid_json(self):
        """tool_calls_json 解析失败时应返回空列表而非抛异常。"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"query_pattern": "水情查询", "tool_calls_json": "not json",
             "success": True, "rounds_used": 1, "use_count": 1},
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("pymysql.connect", return_value=mock_conn):
            result = self.store.get_relevant_skills("水情", limit=3)
        assert len(result) == 1
        assert result[0]["tool_calls"] == []

    def test_add_skill_existing_updates_count(self):
        """同模式技能已存在时应更新 use_count 而非新增。"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (10, 3)  # id=10, use_count=3
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("pymysql.connect", return_value=mock_conn):
            self.store.add_skill("水情查询", [{"tool": "get_hydrology"}], True)

        # 验证执行了 UPDATE 而非 INSERT
        sqls = [call.args[0] for call in mock_cursor.execute.call_args_list]
        assert any("UPDATE agent_skills SET use_count" in s for s in sqls)

    def test_add_reflection_success(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("pymysql.connect", return_value=mock_conn):
            self.store.add_reflection(
                "龙门站水情",
                "tool_failure",
                "工具调用失败后重试成功",
                tool_calls_summary="get_hydrology failed then succeeded",
                final_answer="龙门站当前流量...",
                memories_created=2,
            )
        # 验证 INSERT 被调用
        sql_args = mock_cursor.execute.call_args[0]
        assert "INSERT INTO agent_reflections" in sql_args[0]
        assert sql_args[1][1] == "tool_failure"
        assert sql_args[1][5] == 2  # memories_created


# ====== 单例与全局函数 ======

class TestSingletonAndGlobals:
    def test_get_memory_store_singleton(self):
        """get_memory_store 应返回同一实例（lru_cache）。"""
        s1 = get_memory_store()
        s2 = get_memory_store()
        assert s1 is s2

    def test_is_memory_enabled_disabled_when_no_password(self):
        """MYSQL_PASSWORD 为空时 is_memory_enabled 返回 False。"""
        with patch("app.core.config.get_settings") as mock_settings:
            mock_settings.return_value.SELF_EVOLUTION_ENABLED = True
            mock_settings.return_value.MYSQL_PASSWORD = ""
            # 清除 lru_cache 以使 mock 生效
            get_memory_store.cache_clear()
            assert is_memory_enabled() is False

    def test_is_memory_enabled_disabled_when_switch_off(self):
        """SELF_EVOLUTION_ENABLED=False 时 is_memory_enabled 返回 False。"""
        with patch("app.core.config.get_settings") as mock_settings:
            mock_settings.return_value.SELF_EVOLUTION_ENABLED = False
            mock_settings.return_value.MYSQL_PASSWORD = "secret"
            get_memory_store.cache_clear()
            assert is_memory_enabled() is False
