"""引用标记过滤测试：answer 中 [N] 只允许对应真实联网搜索引用。

背景：闲聊/技能介绍等无联网搜索路径，模型会模仿引用格式编造 [1] 标注；
研判路径中模型也可能给水情/法规等工具数据标编号，但前端只展示
web_search 来源的引用。展示层必须兜底剥离无效标记。
"""
from agent.utils import CitationMarkerFilter, strip_citation_markers


class TestStripCitationMarkers:
    def test_strip_all_when_no_valid_refs(self):
        """无有效引用（闲聊路径）：全部剥离。"""
        text = "我可以提供四类服务：水情查询[1]、洪水预判[2]、应急建议[1]、等级解读[3]。"
        out = strip_citation_markers(text, None)
        assert out == "我可以提供四类服务：水情查询、洪水预判、应急建议、等级解读。"
        assert "[1]" not in out

    def test_keep_only_valid_refs(self):
        """有有效引用集合：保留命中的编号，剥离其余。"""
        text = "据实时水情流量 3200m³/s[3]，媒体报道上游降雨[1]，法规规定[5]需转移。"
        out = strip_citation_markers(text, {1})
        assert "[1]" in out
        assert "[3]" not in out and "[5]" not in out

    def test_empty_and_edge_cases(self):
        assert strip_citation_markers("", None) == ""
        # 4 位以上数字的方括号不是引用标记，保留
        assert strip_citation_markers("数组下标 a[1234]", {1}) == "数组下标 a[1234]"
        # markdown 链接文本不受影响
        assert strip_citation_markers("见[链接](http://x)", None) == "见[链接](http://x)"

    def test_three_digit_refs(self):
        assert strip_citation_markers("来源[123]", {123}) == "来源[123]"
        assert strip_citation_markers("来源[123]", {12}) == "来源"


class TestCitationMarkerFilterStream:
    def test_marker_split_across_deltas(self):
        """[1] 被 token 流切碎时也能正确剥离（关键回归）。"""
        f = CitationMarkerFilter(valid_ref_ids=None)
        out = ""
        for piece in ["水情查询", "[", "1", "]已到位", "。洪水[", "2", "]预警"]:
            out += f.feed(piece)
        out += f.flush()
        assert out == "水情查询已到位。洪水预警"

    def test_stream_keeps_valid_and_flushes_dangling(self):
        """流式保留有效编号；流结束时残留的半个 '[' 按原文结算。"""
        f = CitationMarkerFilter(valid_ref_ids={7})
        out = ""
        for piece in ["数据A[", "7", "]ok；数据B[", "8", "]；结尾是["]:
            out += f.feed(piece)
        out += f.flush()
        assert out == "数据A[7]ok；数据B；结尾是["

    def test_stream_all_stripped_when_none(self):
        f = CitationMarkerFilter(valid_ref_ids=None)
        out = f.feed("技能一[1]技能二[22]技能三[333]")
        out += f.flush()
        assert out == "技能一技能二技能三"


class TestDirectChatPromptRule:
    def test_direct_chat_prompt_forbids_markers(self):
        """闲聊提示词应包含禁止 [数字] 引用标注的规则。"""
        from agent.prompts import DIRECT_CHAT_PROMPT
        assert "引用标注" in DIRECT_CHAT_PROMPT or "[数字]" in DIRECT_CHAT_PROMPT

    def test_citation_guidance_only_web_search(self):
        """引用规范应限定唯一可引用来源是联网搜索结果。"""
        from agent.prompts import CITATION_GUIDANCE
        assert "联网搜索" in CITATION_GUIDANCE
        assert "web_search" in CITATION_GUIDANCE
