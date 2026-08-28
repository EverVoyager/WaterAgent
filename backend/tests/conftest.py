"""pytest 全局 fixtures。"""
import contextlib
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中（agent 模块在项目根，不在 backend/ 下）
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

from agent.skills.models import SkillCreate
from agent.skills.skill_store import get_skill_store, is_skill_store_enabled


@pytest.fixture(autouse=True)
def _restore_skills_table():
    """测试前后保持 skills 表数据不变（备份 → 测试 → 恢复）。

    避免 pytest 清空生产数据。仅在 MySQL 可用时执行。
    """
    if not is_skill_store_enabled():
        yield
        return

    store = get_skill_store()
    # 备份现有数据（建表失败等由具体测试处理）
    backup = []
    with contextlib.suppress(Exception):
        backup = store.list_skills(enabled_only=False)

    # 测试前清空（隔离）
    with contextlib.suppress(Exception):
        store.delete_all()

    yield

    # 测试后恢复原数据
    try:
        store.delete_all()
        for skill in backup:
            store.create_skill(SkillCreate(
                name=skill.name,
                description=skill.description,
                instructions=skill.instructions,
                tool_names=skill.tool_names,
                enabled=skill.enabled,
            ))
    except Exception:
        pass  # 恢复失败不阻塞测试退出
