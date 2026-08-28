"""Citation Grounding 单元测试。

覆盖：
- _verify_citations：引用原文真实性校验（精确子串 + 空白宽松匹配）
- _build_citations_with_metadata：引用与来源元数据拼合 + 无效引用过滤
"""
from agent.graph.synthesizer_node import (
    _build_citations_with_metadata,
    _verify_citations,
)

# ============ _verify_citations ============

class TestVerifyCitations:
    """引用原文真实性校验。"""

    def test_empty_citations_pass(self):
        """无引用时直接通过。"""
        registry = {1: {"text": "某法规条款原文"}}
        ok, msg = _verify_citations([], registry)
        assert ok is True
        assert msg == ""

    def test_empty_registry_empty_citations_pass(self):
        """无引用且无来源时也通过。"""
        ok, msg = _verify_citations([], {})
        assert ok is True

    def test_exact_substring_match_pass(self):
        """quote 是原文的精确子串 → 通过。"""
        registry = {1: {"text": "防汛预警等级阈值标准：Ⅰ级流量≥5000m³/s"}}
        citations = [{"ref_id": 1, "quote": "Ⅰ级流量≥5000m³/s", "source_type": "threshold"}]
        ok, msg = _verify_citations(citations, registry)
        assert ok is True

    def test_whitespace_normalized_match_pass(self):
        """quote 与原文空白不同但去空白后一致 → 通过。"""
        registry = {1: {"text": "流量 5000\n水位 超保证"}}
        citations = [{"ref_id": 1, "quote": "流量5000水位超保证", "source_type": "hydrology"}]
        ok, msg = _verify_citations(citations, registry)
        assert ok is True

    def test_empty_quote_fails(self):
        """quote 为空 → 失败。"""
        registry = {1: {"text": "原文内容"}}
        citations = [{"ref_id": 1, "quote": "", "source_type": "regulation"}]
        ok, msg = _verify_citations(citations, registry)
        assert ok is False
        assert "quote 为空" in msg

    def test_whitespace_only_quote_fails(self):
        """quote 仅含空白 → 失败（strip 后为空）。"""
        registry = {1: {"text": "原文内容"}}
        citations = [{"ref_id": 1, "quote": "   ", "source_type": "regulation"}]
        ok, msg = _verify_citations(citations, registry)
        assert ok is False
        assert "quote 为空" in msg

    def test_unknown_ref_id_fails(self):
        """ref_id 不在 source_registry 中 → 失败。"""
        registry = {1: {"text": "原文"}}
        citations = [{"ref_id": 99, "quote": "原文", "source_type": "regulation"}]
        ok, msg = _verify_citations(citations, registry)
        assert ok is False
        assert "不存在" in msg

    def test_quote_not_in_source_fails(self):
        """quote 不在原文中（LLM 编造）→ 失败。"""
        registry = {1: {"text": "实际流量为 537m³/s"}}
        citations = [{"ref_id": 1, "quote": "实际流量为 9999m³/s", "source_type": "hydrology"}]
        ok, msg = _verify_citations(citations, registry)
        assert ok is False
        assert "找不到" in msg

    def test_multiple_citations_all_valid_pass(self):
        """多条引用全部有效 → 通过。"""
        registry = {
            1: {"text": "流量 537m³/s"},
            2: {"text": "24h 降雨 75mm"},
        }
        citations = [
            {"ref_id": 1, "quote": "流量 537m³/s", "source_type": "hydrology"},
            {"ref_id": 2, "quote": "24h 降雨 75mm", "source_type": "weather"},
        ]
        ok, msg = _verify_citations(citations, registry)
        assert ok is True

    def test_multiple_citations_one_invalid_fails(self):
        """多条引用中有一条无效 → 失败。"""
        registry = {
            1: {"text": "流量 537m³/s"},
            2: {"text": "24h 降雨 75mm"},
        }
        citations = [
            {"ref_id": 1, "quote": "流量 537m³/s", "source_type": "hydrology"},
            {"ref_id": 2, "quote": "编造的降雨数据 200mm", "source_type": "weather"},
        ]
        ok, msg = _verify_citations(citations, registry)
        assert ok is False
        assert "找不到" in msg
        assert "[2]" in msg

    def test_partial_quote_match_pass(self):
        """quote 是原文的一部分（非全文）→ 通过。"""
        registry = {1: {"text": "根据防洪法第三十二条，各级人民政府应当制定防洪预案。"}}
        citations = [{"ref_id": 1, "quote": "各级人民政府应当制定防洪预案", "source_type": "regulation"}]
        ok, msg = _verify_citations(citations, registry)
        assert ok is True


# ============ _build_citations_with_metadata ============

class TestBuildCitationsWithMetadata:
    """引用与来源元数据拼合 + 无效引用过滤。"""

    def test_valid_citation_with_metadata(self):
        """有效引用 → 拼合来源元数据（web_search 结构）。"""
        registry = {
            1: {
                "text": "吴堡水文站流量持续监测中",
                "source_type": "web_search",
                "title": "黄河水利委员会最新汛情通报",
                "url": "https://www.yrcc.gov.cn/news/flood_report.html",
            },
        }
        raw = [{"ref_id": 1, "quote": "吴堡水文站流量持续监测中", "source_type": "web_search"}]
        result = _build_citations_with_metadata(raw, registry)
        assert len(result) == 1
        cite = result[0]
        assert cite["ref_id"] == 1
        assert cite["quote"] == "吴堡水文站流量持续监测中"
        assert cite["title"] == "黄河水利委员会最新汛情通报"
        assert cite["url"] == "https://www.yrcc.gov.cn/news/flood_report.html"
        assert cite["source_type"] == "web_search"

    def test_filters_invalid_quote(self):
        """quote 不在原文中 → 过滤掉。"""
        registry = {1: {"text": "实际数据", "source_type": "web_search",
                         "title": "站", "url": "https://example.com"}}
        raw = [{"ref_id": 1, "quote": "编造数据", "source_type": "web_search"}]
        result = _build_citations_with_metadata(raw, registry)
        assert len(result) == 0

    def test_filters_unknown_ref_id(self):
        """ref_id 不在 registry 中 → 过滤掉。"""
        registry = {1: {"text": "原文", "source_type": "web_search",
                         "title": "", "url": ""}}
        raw = [{"ref_id": 99, "quote": "原文", "source_type": "web_search"}]
        result = _build_citations_with_metadata(raw, registry)
        assert len(result) == 0

    def test_filters_empty_quote(self):
        """quote 为空 → 过滤掉。"""
        registry = {1: {"text": "原文", "source_type": "web_search",
                         "title": "", "url": ""}}
        raw = [{"ref_id": 1, "quote": "", "source_type": "web_search"}]
        result = _build_citations_with_metadata(raw, registry)
        assert len(result) == 0

    def test_deduplicates_by_ref_id(self):
        """同一 ref_id 多次引用 → 只保留一条。"""
        registry = {1: {"text": "原文片段", "source_type": "web_search",
                         "title": "网页A", "url": "https://a.com"}}
        raw = [
            {"ref_id": 1, "quote": "原文片段", "source_type": "web_search"},
            {"ref_id": 1, "quote": "原文片段", "source_type": "web_search"},
        ]
        result = _build_citations_with_metadata(raw, registry)
        assert len(result) == 1

    def test_source_type_always_from_registry(self):
        """source_type 始终取自 registry（只有 web_search 可引用）。"""
        registry = {1: {"text": "原文", "source_type": "web_search",
                         "title": "网页", "url": "https://example.com"}}
        raw = [{"ref_id": 1, "quote": "原文", "source_type": "web_search"}]
        result = _build_citations_with_metadata(raw, registry)
        assert len(result) == 1
        assert result[0]["source_type"] == "web_search"

    def test_mixed_valid_and_invalid(self):
        """有效与无效引用混合 → 只保留有效的。"""
        registry = {
            1: {"text": "有效原文", "source_type": "web_search",
                 "title": "网页A", "url": "https://a.com"},
            2: {"text": "另一段原文", "source_type": "web_search",
                 "title": "网页B", "url": "https://b.com"},
        }
        raw = [
            {"ref_id": 1, "quote": "有效原文", "source_type": "web_search"},
            {"ref_id": 2, "quote": "编造内容", "source_type": "web_search"},
            {"ref_id": 99, "quote": "未知编号", "source_type": "web_search"},
        ]
        result = _build_citations_with_metadata(raw, registry)
        assert len(result) == 1
        assert result[0]["ref_id"] == 1
        assert result[0]["title"] == "网页A"

    def test_empty_raw_returns_empty(self):
        """空引用列表 → 空结果。"""
        registry = {1: {"text": "原文", "source_type": "web_search",
                         "title": "", "url": ""}}
        assert _build_citations_with_metadata([], registry) == []

    def test_whitespace_normalized_quote_preserved(self):
        """空白宽松匹配通过的 quote → 保留（原文 quote 不做修改）。"""
        registry = {1: {"text": "流量 5000\n水位 超保证",
                         "source_type": "web_search", "title": "网页",
                         "url": "https://example.com"}}
        raw = [{"ref_id": 1, "quote": "流量5000水位超保证", "source_type": "web_search"}]
        result = _build_citations_with_metadata(raw, registry)
        assert len(result) == 1
        assert result[0]["quote"] == "流量5000水位超保证"
