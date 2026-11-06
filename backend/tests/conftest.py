"""pytest 全局 fixtures。"""
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中（agent 模块在项目根，不在 backend/ 下）
for _path in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pytest

# 必须在 Settings 首次加载前切换，测试绝不备份/清空业务库。
os.environ["MYSQL_DATABASE"] = os.environ.get("TEST_MYSQL_DATABASE", "water_agent_test")

from agent.skills.skill_store import get_skill_store, is_skill_store_enabled
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def memory_qdrant(monkeypatch, request):
    """单元测试只操作独立内存库，禁止触碰实际的记忆正文集合。"""
    if request.node.get_closest_marker("integration"):
        yield None
        return
    from qdrant_client import QdrantClient

    from agent.memory import vector_index

    client = QdrantClient(":memory:")
    monkeypatch.setattr(vector_index, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(vector_index, "get_qdrant_config", lambda: {"vector_size": 3})
    yield client
    client.close()


@pytest.fixture(autouse=True)
def _restore_skills_table():
    """只清理专用测试库，数据库故障立即报错。"""
    if not is_skill_store_enabled():
        yield
        return

    if not get_settings().MYSQL_DATABASE.endswith("_test"):
        pytest.fail("测试数据库名称必须以 _test 结尾，拒绝清理业务库")
    store = get_skill_store()
    store.delete_all()
    try:
        yield
    finally:
        store.delete_all()
