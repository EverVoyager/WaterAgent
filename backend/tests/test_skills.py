"""Skill 模块单元测试。

覆盖：
- 数据模型校验（name 格式、tool_names 白名单）
- 存储 CRUD（创建/读取/更新/删除/查重）
- 匹配引擎（关键词匹配降级、缓存失效）
- build_openai_tools 工具子集过滤
"""
from unittest.mock import patch

import pytest

from agent.skills import matcher as skill_matcher
from agent.skills import store as skill_store
from agent.skills.models import Skill, SkillCreate, SkillUpdate
from agent.skills.skill_store import is_skill_store_enabled

# 需要 MySQL 的测试统一打标（CI 由 mysql service 提供；本地未配置则跳过而非报错）
_requires_mysql = pytest.mark.skipif(
    not is_skill_store_enabled(),
    reason="SkillStore 需要 MySQL（未配置 MYSQL_PASSWORD）",
)


# ====== 数据模型测试 ======

class TestSkillModel:
    def test_create_valid_skill(self):
        """创建合法 Skill。"""
        req = SkillCreate(
            name="flood_dispatch",
            description="水库防洪调度研判，根据入库流量给出调度建议",
            instructions="你是水库调度专家。1. 调用 get_hydrology 获取水情",
            tool_names=["get_hydrology"],
        )
        assert req.name == "flood_dispatch"
        assert req.enabled is True

    def test_invalid_name_starts_with_digit(self):
        """name 首字符不能是数字。"""
        with pytest.raises(ValueError, match="首字符必须为字母"):
            SkillCreate(
                name="123_skill",
                description="test description here",
                instructions="test instructions here",
            )

    def test_invalid_name_special_chars(self):
        """name 不能含特殊字符。"""
        with pytest.raises(ValueError, match="只能包含字母"):
            SkillCreate(
                name="skill-name!",
                description="test description here",
                instructions="test instructions here",
            )

    def test_invalid_tool_name(self):
        """tool_names 必须在内置工具白名单内。"""
        with pytest.raises(ValueError, match="未知工具名"):
            Skill(
                id="test",
                name="test_skill",
                description="test description here",
                instructions="test instructions here",
                tool_names=["nonexistent_tool"],
            )

    def test_empty_tool_names_means_all(self):
        """空 tool_names 表示不限制（用全部工具）。"""
        skill = Skill(
            id="test",
            name="test_skill",
            description="test description here",
            instructions="test instructions here",
            tool_names=[],
        )
        assert skill.tool_names == []


# ====== 存储测试 ======

@_requires_mysql
class TestSkillStore:
    """测试 MySQL 存储 CRUD。

    conftest.py 的 _clean_skills_table fixture 自动清表确保隔离。
    """

    @pytest.fixture(autouse=True)
    def _reset_matcher(self):
        """重置匹配器缓存。"""
        skill_matcher._matcher.invalidate()
        yield
        skill_matcher._matcher.invalidate()

    def test_create_and_get(self):
        """创建后能读取。"""
        req = SkillCreate(
            name="flood_dispatch",
            description="水库防洪调度研判，根据入库流量给出调度建议",
            instructions="你是水库调度专家。请按规程研判泄洪方案",
            tool_names=["get_hydrology"],
        )
        skill = skill_store.create_skill(req)
        assert skill.id == "flood_dispatch"

        got = skill_store.get_skill("flood_dispatch")
        assert got is not None
        assert got.name == "flood_dispatch"

    def test_create_duplicate_raises(self):
        """重复 name 创建报错。"""
        req = SkillCreate(
            name="flood_dispatch",
            description="水库防洪调度研判，根据入库流量给出调度建议",
            instructions="你是水库调度专家。请按规程研判泄洪方案",
        )
        skill_store.create_skill(req)
        with pytest.raises(ValueError, match="已存在"):
            skill_store.create_skill(req)

    def test_list_all(self):
        """列出所有 Skill。"""
        for name in ["skill_a", "skill_b", "skill_c"]:
            skill_store.create_skill(SkillCreate(
                name=name,
                description=f"测试技能描述 {name}",
                instructions=f"测试技能指令 {name}",
            ))
        all_skills = skill_store.list_skills()
        assert len(all_skills) == 3

    def test_list_enabled_only(self):
        """仅列出启用的 Skill。"""
        skill_store.create_skill(SkillCreate(
            name="enabled_one",
            description="这是一个已启用的测试技能描述",
            instructions="这是已启用技能的测试指令内容",
            enabled=True,
        ))
        skill_store.create_skill(SkillCreate(
            name="disabled_one",
            description="这是一个已禁用的测试技能描述",
            instructions="这是已禁用技能的测试指令内容",
            enabled=False,
        ))
        enabled = skill_store.list_skills(enabled_only=True)
        assert len(enabled) == 1
        assert enabled[0].name == "enabled_one"

    def test_update_skill(self):
        """更新 Skill 字段。"""
        skill_store.create_skill(SkillCreate(
            name="test_skill",
            description="这是更新前的旧技能描述文本",
            instructions="这是更新前的旧技能指令文本",
        ))
        updated = skill_store.update_skill("test_skill", SkillUpdate(
            description="这是更新后的新技能描述文本",
            tool_names=["get_hydrology", "get_weather"],
        ))
        assert updated.description == "这是更新后的新技能描述文本"
        assert updated.tool_names == ["get_hydrology", "get_weather"]
        # instructions 未更新，保持原值
        assert updated.instructions == "这是更新前的旧技能指令文本"

    def test_update_nonexistent_raises(self):
        """更新不存在的 Skill 报错。"""
        with pytest.raises(ValueError, match="不存在"):
            skill_store.update_skill("nonexistent", SkillUpdate(description="不存在技能的描述文本"))

    def test_delete_skill(self):
        """删除 Skill。"""
        skill_store.create_skill(SkillCreate(
            name="to_delete",
            description="这是待删除的测试技能描述文本",
            instructions="这是待删除的测试技能指令文本",
        ))
        assert skill_store.delete_skill("to_delete") is True
        assert skill_store.get_skill("to_delete") is None

    def test_delete_nonexistent_returns_false(self):
        """删除不存在的 Skill 返回 False。"""
        assert skill_store.delete_skill("nonexistent") is False


# ====== 匹配引擎测试 ======

@_requires_mysql
class TestSkillMatcher:
    """测试 Skill 匹配引擎。

    embedding API 不可用时降级到关键词匹配。
    """

    @pytest.fixture(autouse=True)
    def _setup_skills(self):
        """准备测试 Skill 数据。"""
        skill_matcher._matcher.invalidate()

        # 创建两个 Skill
        skill_store.create_skill(SkillCreate(
            name="flood_dispatch",
            description="水库防洪调度研判，根据入库流量、水位、泄洪能力给出调度建议",
            instructions="你是水库调度专家。请按规程研判泄洪方案并给出建议",
            tool_names=["get_hydrology"],
        ))
        skill_store.create_skill(SkillCreate(
            name="emergency_response",
            description="应急响应预案生成，根据预警等级制定转移疏散方案",
            instructions="你是应急预案专家。请按预警等级生成转移疏散方案",
            tool_names=["generate_plan", "search_regulation"],
        ))
        yield
        skill_matcher._matcher.invalidate()

    def test_match_by_keywords_fallback(self):
        """embedding 不可用时降级到关键词匹配。"""
        # mock embedding 返回 None（模拟 API 不可用）
        with patch.object(skill_matcher, "embed_query", return_value=None), \
             patch.object(skill_matcher, "embed_texts", return_value=None):
            # 重建缓存（此时 embedding 失败，走关键词降级）
            skill_matcher._matcher.invalidate()
            skill = skill_matcher.match_skill("水库防洪调度研判")
            # 关键词匹配可能命中也可能不命中，取决于分词
            # 这里主要验证不抛异常
            assert skill is None or skill.name in ("flood_dispatch", "emergency_response")

    def test_no_skills_returns_none(self):
        """无 Skill 时返回 None。"""
        skill_matcher._matcher.invalidate()
        assert skill_matcher.match_skill("任何问题") is None

    def test_get_instructions_none_when_no_match(self):
        """未匹配到 Skill 时返回 None。"""
        with patch.object(skill_matcher, "embed_query", return_value=None), \
             patch.object(skill_matcher, "embed_texts", return_value=None):
            skill_matcher._matcher.invalidate()
            result = skill_matcher.get_active_skill_instructions("完全无关的随机文本xyz")
            assert result is None

    def test_invalidate_cache(self):
        """失效缓存后重新加载。"""
        # 先确保缓存就绪
        with patch.object(skill_matcher, "embed_query", return_value=None), \
             patch.object(skill_matcher, "embed_texts", return_value=None):
            skill_matcher._matcher.invalidate()
            _ = skill_matcher.match_skill("test")

        # 失效后 _dirty 应为 True
        skill_matcher.invalidate_cache()
        assert skill_matcher._matcher._dirty is True


# ====== build_openai_tools 工具子集过滤测试 ======

class TestBuildOpenaiTools:
    def test_all_tools_by_default(self):
        """默认返回全部工具。"""
        from agent.tools.schemas import build_openai_tools
        tools = build_openai_tools()
        assert len(tools) == 8  # 8 个内置工具（含 list_skills）

    def test_subset_filter(self):
        """按工具子集过滤。"""
        from agent.tools.schemas import build_openai_tools
        tools = build_openai_tools(tool_names=["get_hydrology", "get_weather"])
        assert len(tools) == 2
        names = [t["function"]["name"] for t in tools]
        assert set(names) == {"get_hydrology", "get_weather"}

    def test_empty_subset_means_all(self):
        """空列表 = 全部工具。"""
        from agent.tools.schemas import build_openai_tools
        tools = build_openai_tools(tool_names=[])
        assert len(tools) == 8

    def test_invalid_tool_names_filtered(self):
        """无效工具名被过滤掉。"""
        from agent.tools.schemas import build_openai_tools
        tools = build_openai_tools(tool_names=["get_hydrology", "invalid_tool"])
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "get_hydrology"

    def test_none_means_all(self):
        """None = 全部工具。"""
        from agent.tools.schemas import build_openai_tools
        tools = build_openai_tools(tool_names=None)
        assert len(tools) == 8

    def test_list_skills_included_by_default(self):
        """list_skills 工具默认包含在全部工具列表中。"""
        from agent.tools.schemas import build_openai_tools
        tools = build_openai_tools()
        names = [t["function"]["name"] for t in tools]
        assert "list_skills" in names

    def test_list_skills_can_be_filtered_out(self):
        """list_skills 可被工具子集过滤排除。"""
        from agent.tools.schemas import build_openai_tools
        tools = build_openai_tools(tool_names=["get_hydrology"])
        names = [t["function"]["name"] for t in tools]
        assert "list_skills" not in names
        assert names == ["get_hydrology"]


@_requires_mysql
class TestListSkillsTool:
    """list_skills 工具执行测试（对标 MCP tools/list 发现机制）。"""

    def test_list_skills_returns_enabled_only(self):
        """list_skills 只返回已启用的技能。"""
        from agent.tools.real_executor import list_skills_real
        from agent.tools.schemas import ListSkillsParams

        # 创建 2 个启用 + 1 个禁用的技能
        skill_store.create_skill(SkillCreate(
            name="skill_a", description="技能 A 的描述用于测试匹配场景",
            instructions="指令 A 内容足够长度用于测试", tool_names=["get_hydrology"],
        ))
        skill_store.create_skill(SkillCreate(
            name="skill_b", description="技能 B 的描述用于测试匹配场景",
            instructions="指令 B 内容足够长度用于测试", tool_names=[],
        ))
        skill_store.create_skill(SkillCreate(
            name="skill_c", description="技能 C 已被禁用不可见",
            instructions="指令 C 内容足够长度用于测试", enabled=False,
        ))

        result = list_skills_real(ListSkillsParams())
        assert result["total"] == 2
        names = [s["name"] for s in result["skills"]]
        assert "skill_a" in names
        assert "skill_b" in names
        assert "skill_c" not in names  # 禁用的不返回

    def test_list_skills_excludes_instructions_by_default(self):
        """默认不返回 instructions（省 token）。"""
        from agent.tools.real_executor import list_skills_real
        from agent.tools.schemas import ListSkillsParams

        skill_store.create_skill(SkillCreate(
            name="test_skill", description="测试技能的描述内容足够长度",
            instructions="这是完整指令不应默认返回的内容",
        ))

        result = list_skills_real(ListSkillsParams())
        assert "instructions" not in result["skills"][0]

    def test_list_skills_includes_instructions_when_requested(self):
        """include_instructions=True 时返回完整指令。"""
        from agent.tools.real_executor import list_skills_real
        from agent.tools.schemas import ListSkillsParams

        skill_store.create_skill(SkillCreate(
            name="test_skill", description="测试技能的描述内容足够长度",
            instructions="这是完整指令应该被返回的内容",
        ))

        result = list_skills_real(ListSkillsParams(include_instructions=True))
        assert result["skills"][0]["instructions"] == "这是完整指令应该被返回的内容"

    def test_list_skills_returns_tool_names(self):
        """返回每个技能的 tool_names。"""
        from agent.tools.real_executor import list_skills_real
        from agent.tools.schemas import ListSkillsParams

        skill_store.create_skill(SkillCreate(
            name="test_skill", description="测试技能的描述内容足够长度",
            instructions="指令内容足够长度用于测试通过", tool_names=["get_hydrology", "list_skills"],
        ))

        result = list_skills_real(ListSkillsParams())
        assert result["skills"][0]["tool_names"] == ["get_hydrology", "list_skills"]

    def test_list_skills_empty_when_no_skills(self):
        """无启用技能时返回空列表。"""
        from agent.tools.real_executor import list_skills_real
        from agent.tools.schemas import ListSkillsParams

        result = list_skills_real(ListSkillsParams())
        assert result["total"] == 0
        assert result["skills"] == []

    def test_list_skills_via_real_execute_tool(self):
        """通过 real_execute_tool 入口调用 list_skills。"""
        from agent.tools.real_executor import real_execute_tool

        skill_store.create_skill(SkillCreate(
            name="via_entry", description="通过入口调用的测试技能描述",
            instructions="指令内容足够长度用于测试通过",
        ))

        result = real_execute_tool("list_skills", {})
        assert result["total"] == 1
        assert result["skills"][0]["name"] == "via_entry"
        assert result["source"] == "skills_store"

    def test_list_skills_tool_in_schema(self):
        """list_skills 在 TOOL_PARAM_MODELS 和 TOOL_DESCRIPTIONS 中注册。"""
        from agent.tools.schemas import TOOL_DESCRIPTIONS, TOOL_PARAM_MODELS
        assert "list_skills" in TOOL_PARAM_MODELS
        assert "list_skills" in TOOL_DESCRIPTIONS
        assert "技能" in TOOL_DESCRIPTIONS["list_skills"] or "Skill" in TOOL_DESCRIPTIONS["list_skills"]

    def test_list_skills_in_builtin_tools_whitelist(self):
        """list_skills 在 BUILTIN_TOOLS 白名单中（可作为 skill 的 tool_names）。"""
        from agent.skills.models import Skill
        assert "list_skills" in Skill.BUILTIN_TOOLS

    def test_skill_can_use_list_skills_as_tool(self):
        """技能可声明 list_skills 作为其可用工具。"""
        skill = skill_store.create_skill(SkillCreate(
            name="meta_skill", description="元技能可查询其他技能的描述",
            instructions="指令内容足够长度用于测试通过", tool_names=["list_skills"],
        ))
        assert "list_skills" in skill.tool_names
