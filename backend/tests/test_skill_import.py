"""Skill 包导入功能单元测试。

覆盖：
- SKILL.md 解析器（frontmatter + 正文）
- allowed-tools 白名单过滤
- 包导入逻辑（.md / .zip 两种格式）
- 冲突处理（overwrite / rename / cancel）
- 安全校验（ZIP 炸弹 / 路径穿越 / 大小限制）
"""
import io
import zipfile

import pytest

from agent.skills import matcher as skill_matcher
from agent.skills.importer import (
    _find_skill_md,
    _validate_zip_safety,
    import_skill_from_md,
    import_skill_from_zip,
)
from agent.skills.parser import (
    extract_tool_name,
    filter_tools_by_whitelist,
    parse_skill_md,
)
from agent.skills.skill_store import is_skill_store_enabled

# 需要落库的导入测试统一打标（解析/过滤/安全校验等纯函数测试不受影响）
_requires_mysql = pytest.mark.skipif(
    not is_skill_store_enabled(),
    reason="SkillStore 需要 MySQL（未配置 MYSQL_PASSWORD）",
)


# ====== 解析器测试 ======

class TestParseSkillMd:
    def test_parse_standard_format(self):
        """标准 SKILL.md 格式解析。"""
        content = """---
name: flood_dispatch
description: 水库防洪调度研判，根据入库流量给出建议
license: MIT
allowed-tools: get_hydrology get_weather
---
# 防洪调度技能

当用户询问水库调度时，按以下步骤执行：
1. 调用 get_hydrology 获取水情
2. 基于规程研判泄洪方案
"""
        result = parse_skill_md(content)
        assert result.name == "flood_dispatch"
        assert "水库防洪调度" in result.description
        assert "防洪调度技能" in result.instructions
        assert result.raw_tool_names == ["get_hydrology", "get_weather"]
        assert result.license == "MIT"

    def test_parse_list_format_allowed_tools(self):
        """allowed-tools 支持 YAML 列表格式。"""
        content = """---
name: test_skill
description: 这是一个测试技能描述文本
allowed-tools:
  - get_hydrology
  - get_weather
---
测试指令内容足够长度
"""
        result = parse_skill_md(content)
        assert result.raw_tool_names == ["get_hydrology", "get_weather"]

    def test_parse_claude_tool_format(self):
        """兼容 Claude 的 Bash(python:*) 格式。"""
        content = """---
name: test_skill
description: 这是一个测试技能描述文本
allowed-tools: Bash(python:*) Read get_hydrology
---
测试指令内容足够长度
"""
        result = parse_skill_md(content)
        assert result.raw_tool_names == ["Bash(python:*)", "Read", "get_hydrology"]

    def test_parse_no_allowed_tools(self):
        """无 allowed-tools 字段时 raw_tool_names 为空。"""
        content = """---
name: test_skill
description: 这是一个测试技能描述文本
---
测试指令内容足够长度
"""
        result = parse_skill_md(content)
        assert result.raw_tool_names == []

    def test_parse_no_body_uses_description(self):
        """frontmatter 后无正文时，用 description 兜底。"""
        content = """---
name: test_skill
description: 这是一个测试技能描述文本
---
"""
        result = parse_skill_md(content)
        assert result.instructions == "这是一个测试技能描述文本"

    def test_parse_missing_frontmatter_raises(self):
        """缺少 frontmatter 报错。"""
        with pytest.raises(ValueError, match="frontmatter"):
            parse_skill_md("# Just markdown\nNo frontmatter")

    def test_parse_missing_name_raises(self):
        """缺少 name 字段报错。"""
        content = """---
description: 这是一个测试技能描述文本
---
正文
"""
        with pytest.raises(ValueError, match="name"):
            parse_skill_md(content)

    def test_parse_missing_description_raises(self):
        """缺少 description 字段报错。"""
        content = """---
name: test_skill
---
正文
"""
        with pytest.raises(ValueError, match="description"):
            parse_skill_md(content)

    def test_parse_empty_content_raises(self):
        """空内容报错。"""
        with pytest.raises(ValueError, match="为空"):
            parse_skill_md("")

    def test_parse_invalid_yaml_raises(self):
        """YAML 格式错误报错。"""
        content = """---
name: [unclosed
description: test
---
正文
"""
        with pytest.raises(ValueError, match="YAML"):
            parse_skill_md(content)


# ====== 工具名过滤测试 ======

class TestToolFiltering:
    def test_extract_tool_name_plain(self):
        """普通工具名提取。"""
        assert extract_tool_name("get_hydrology") == "get_hydrology"

    def test_extract_tool_name_claude_format(self):
        """Claude 格式工具名提取（去括号 + 转小写）。"""
        assert extract_tool_name("Bash(python:*)") == "bash"
        assert extract_tool_name("Read") == "read"

    def test_filter_keeps_builtin_tools(self):
        """白名单内工具保留。"""
        builtin = {"get_hydrology", "get_weather"}
        filtered, warnings = filter_tools_by_whitelist(
            ["get_hydrology", "get_weather"], builtin
        )
        assert filtered == ["get_hydrology", "get_weather"]
        assert warnings == []

    def test_filter_drops_unknown_tools(self):
        """非白名单工具被丢弃并告警。"""
        builtin = {"get_hydrology"}
        filtered, warnings = filter_tools_by_whitelist(
            ["get_hydrology", "Bash(python:*)", "unknown_tool"], builtin
        )
        assert filtered == ["get_hydrology"]
        assert len(warnings) == 2
        assert any("Bash" in w for w in warnings)
        assert any("unknown_tool" in w for w in warnings)

    def test_filter_empty_input(self):
        """空输入返回空。"""
        filtered, warnings = filter_tools_by_whitelist([], {"get_hydrology"})
        assert filtered == []
        assert warnings == []


# ====== 导入逻辑测试 ======

@_requires_mysql
class TestImportFromMd:
    """从 SKILL.md 文本导入。"""

    @pytest.fixture(autouse=True)
    def _reset_matcher(self):
        """重置匹配器缓存（conftest 自动清表）。"""
        skill_matcher._matcher.invalidate()
        yield
        skill_matcher._matcher.invalidate()

    def test_import_creates_new_skill(self):
        """成功导入新技能。"""
        content = """---
name: flood_dispatch
description: 水库防洪调度研判，根据入库流量给出建议
allowed-tools: get_hydrology get_weather
---
你是水库调度专家。请按规程研判泄洪方案。
"""
        result = import_skill_from_md(content)
        assert result.action == "created"
        assert result.final_name == "flood_dispatch"
        assert result.skill is not None
        assert result.skill.tool_names == ["get_hydrology", "get_weather"]

    def test_import_with_claude_tools_filtered(self):
        """Claude 工具名被过滤，仅保留白名单内工具。"""
        content = """---
name: test_skill
description: 这是一个测试技能描述文本
allowed-tools: Bash(python:*) Read get_hydrology
---
测试指令内容足够长度
"""
        result = import_skill_from_md(content)
        assert result.skill.tool_names == ["get_hydrology"]
        assert len(result.warnings) == 2  # Bash 和 Read 被过滤

    def test_import_no_allowed_tools_means_all(self):
        """无 allowed-tools 时 tool_names 为空（= 全部工具）。"""
        content = """---
name: test_skill
description: 这是一个测试技能描述文本
---
测试指令内容足够长度
"""
        result = import_skill_from_md(content)
        assert result.skill.tool_names == []

    def test_import_hyphen_name_normalized(self):
        """Claude 连字符 name 自动转下划线。"""
        content = """---
name: flood-dispatch
description: 这是一个测试技能描述文本
---
测试指令内容足够长度
"""
        result = import_skill_from_md(content)
        assert result.final_name == "flood_dispatch"
        assert result.original_name == "flood_dispatch"

    def test_import_conflict_cancel_raises(self):
        """冲突策略 cancel 时报错。"""
        content = """---
name: test_skill
description: 这是一个测试技能描述文本
---
测试指令内容足够长度
"""
        import_skill_from_md(content)  # 第一次导入
        with pytest.raises(ValueError, match="已存在"):
            import_skill_from_md(content, conflict_strategy="cancel")

    def test_import_conflict_overwrite(self):
        """冲突策略 overwrite 覆盖原技能。"""
        content = """---
name: test_skill
description: 这是一个测试技能描述文本
---
测试指令内容足够长度
"""
        import_skill_from_md(content)
        result = import_skill_from_md(content, conflict_strategy="overwrite")
        assert result.action == "overwritten"
        assert result.final_name == "test_skill"

    def test_import_conflict_rename(self):
        """冲突策略 rename 自动加后缀。"""
        content = """---
name: test_skill
description: 这是一个测试技能描述文本
---
测试指令内容足够长度
"""
        import_skill_from_md(content)
        result = import_skill_from_md(content, conflict_strategy="rename")
        assert result.action == "renamed"
        assert result.final_name == "test_skill_imported_1"

    def test_import_invalid_strategy_raises(self):
        """无效冲突策略报错。"""
        content = """---
name: test_skill
description: 这是一个测试技能描述文本
---
测试指令内容足够长度
"""
        with pytest.raises(ValueError, match="无效的冲突策略"):
            import_skill_from_md(content, conflict_strategy="invalid")


# ====== ZIP 导入测试 ======

@_requires_mysql
class TestImportFromZip:
    """从 ZIP 包导入。"""

    @pytest.fixture(autouse=True)
    def _reset_matcher(self):
        """重置匹配器缓存（conftest 自动清表）。"""
        skill_matcher._matcher.invalidate()
        yield
        skill_matcher._matcher.invalidate()

    def _make_zip(self, files: dict) -> bytes:
        """构建 ZIP 文件。files = {filename: content_str_or_bytes}"""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, data in files.items():
                if isinstance(data, str):
                    data = data.encode("utf-8")
                zf.writestr(name, data)
        return buf.getvalue()

    def test_import_zip_with_root_skill_md(self):
        """ZIP 根目录含 SKILL.md。"""
        skill_md = """---
name: zip_skill
description: 这是一个 ZIP 导入的测试技能
---
ZIP 导入测试指令内容足够长度
"""
        zip_bytes = self._make_zip({"SKILL.md": skill_md})
        result = import_skill_from_zip(zip_bytes)
        assert result.action == "created"
        assert result.final_name == "zip_skill"

    def test_import_zip_with_nested_skill_md(self):
        """ZIP 子目录含 SKILL.md。"""
        skill_md = """---
name: nested_skill
description: 这是一个嵌套目录的测试技能
---
嵌套目录测试指令内容足够长度
"""
        zip_bytes = self._make_zip({"my-skill/SKILL.md": skill_md})
        result = import_skill_from_zip(zip_bytes)
        assert result.final_name == "nested_skill"

    def test_import_zip_case_insensitive_filename(self):
        """SKILL.md 文件名不区分大小写。"""
        skill_md = """---
name: case_skill
description: 这是一个大小写测试技能
---
大小写测试指令内容足够长度
"""
        zip_bytes = self._make_zip({"skill.md": skill_md})
        result = import_skill_from_zip(zip_bytes)
        assert result.final_name == "case_skill"

    def test_import_zip_no_skill_md_raises(self):
        """ZIP 无 SKILL.md 报错。"""
        zip_bytes = self._make_zip({"README.md": b"no skill here"})
        with pytest.raises(ValueError, match="未找到 SKILL.md"):
            import_skill_from_zip(zip_bytes)

    def test_import_zip_bad_zip_raises(self):
        """无效 ZIP 报错。"""
        with pytest.raises(ValueError, match="无效的 ZIP"):
            import_skill_from_zip(b"not a zip file")

    def test_import_zip_path_traversal_rejected(self):
        """路径穿越被拦截。"""
        skill_md = """---
name: evil_skill
description: 这是一个路径穿越测试技能
---
路径穿越测试指令内容足够长度
"""
        zip_bytes = self._make_zip({"../escape/SKILL.md": skill_md})
        with pytest.raises(ValueError, match="非法路径"):
            import_skill_from_zip(zip_bytes)

    def test_import_zip_absolute_path_rejected(self):
        """绝对路径被拦截。"""
        skill_md = """---
name: abs_skill
description: 这是一个绝对路径测试技能
---
绝对路径测试指令内容足够长度
"""
        # 手动构建含绝对路径的 ZIP
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("/etc/SKILL.md", skill_md.encode("utf-8"))
        with pytest.raises(ValueError, match="非法路径"):
            import_skill_from_zip(buf.getvalue())


# ====== ZIP 安全校验测试 ======

class TestZipSafety:
    def test_validate_normal_zip_passes(self):
        """正常 ZIP 通过校验。"""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("SKILL.md", b"content")
        zf = zipfile.ZipFile(buf)
        _validate_zip_safety(zf)  # 不抛异常即通过

    def test_validate_path_traversal_rejected(self):
        """路径穿越被拦截。"""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../escape.txt", b"content")
        zf = zipfile.ZipFile(buf)
        with pytest.raises(ValueError, match="非法路径"):
            _validate_zip_safety(zf)

    def test_find_skill_md_returns_shortest_path(self):
        """多个 SKILL.md 时返回路径最短的。"""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("SKILL.md", b"root")
            zf.writestr("subdir/SKILL.md", b"nested")
        zf = zipfile.ZipFile(buf)
        path = _find_skill_md(zf)
        assert path == "SKILL.md"

    def test_find_skill_md_none_when_absent(self):
        """无 SKILL.md 返回 None。"""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("README.md", b"content")
        zf = zipfile.ZipFile(buf)
        assert _find_skill_md(zf) is None
