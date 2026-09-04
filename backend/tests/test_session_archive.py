"""会话任务段落盘与按需还原测试（Context-Folding 工程化）。

锁定 KV Cache 三不变量与核心机制：
1. 摘要冻结：生成一次后永不改写，段追加只追加"续摘要"
2. 摘要确定性靠文件固化：同段所有请求读同一份冻结文本
3. 段边界只增不变：新轮次不改旧边界

其余：语义分段、归档幂等（含工具数据覆盖）、压缩产出、按需还原匹配、
压缩标记双兼容。
"""
from unittest.mock import MagicMock, patch

import pytest

from agent.memory import session_archive as sa
from app.core.config import get_settings

# ============ 测试基础设施 ============

def _topic_vec(topic: str) -> list[float]:
    """按主题返回 3 维正交向量（水情/天气/预案）。"""
    return {
        "hydro": [1.0, 0.0, 0.0],
        "weather": [0.0, 1.0, 0.0],
        "plan": [0.0, 0.0, 1.0],
    }[topic]


def _fake_embed_texts(texts):
    """按关键词映射主题向量的 embedding mock。"""
    vecs = []
    for t in texts:
        if any(k in t for k in ("水情", "流量", "水位", "吴堡")):
            vecs.append(_topic_vec("hydro"))
        elif any(k in t for k in ("天气", "降雨", "气温")):
            vecs.append(_topic_vec("weather"))
        elif any(k in t for k in ("预案", "响应", "转移")):
            vecs.append(_topic_vec("plan"))
        else:
            vecs.append([0.5, 0.5, 0.5])
    import numpy as np
    return np.array(vecs, dtype="float32")


def _round_history(*pairs) -> list[dict]:
    """构造 user/assistant 交替 history。pairs = [(query, answer), ...]"""
    msgs = []
    for q, a in pairs:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    return msgs


@pytest.fixture()
def archive_env(tmp_path, monkeypatch):
    """归档目录指向 tmp_path + 可控 embedding mock。"""
    monkeypatch.setenv("SESSION_ARCHIVE_DIR", str(tmp_path / "session_archive"))
    monkeypatch.setenv("SESSION_ARCHIVE_ENABLED", "true")
    get_settings.cache_clear()
    sa._EMB_CACHE.clear()

    def _fake(query):
        return [list(map(float, v)) for v in _fake_embed_texts(query)]

    with patch("agent.rag.embedding.embed_texts", side_effect=_fake):
        yield tmp_path / "session_archive"

    get_settings.cache_clear()
    sa._EMB_CACHE.clear()


def _mock_summary_llm(counter, texts=None):
    """段摘要 LLM mock：记录调用次数，返回固定递增文本（冻结可验证）。"""
    def _client():
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        counter["n"] += 1
        n = counter["n"]
        resp.choices[0].message.content = f"[历史任务] 意图：mock摘要#{n}｜结论：x"
        client.with_options.return_value.chat.completions.create.return_value = resp
        return client

    return _client


# ============ 分段 ============

class TestSegmentation:
    def test_semantic_boundary_splits(self, archive_env):
        rounds = sa.extract_rounds(_round_history(
            ("吴堡站水情如何", "流量 537"),
            ("吴堡站水位多少", "水位 636"),
            ("吕梁天气怎么样", "晴 26 度"),
            ("明天降雨大吗", "小雨"),
        ))
        segs = sa.segment_rounds(rounds)
        assert len(segs) == 2  # 水情段 + 天气段
        assert len(segs[0].rounds) == 2
        assert len(segs[1].rounds) == 2

    def test_same_topic_single_segment(self, archive_env):
        rounds = sa.extract_rounds(_round_history(
            ("吴堡站水情如何", "流量 537"),
            ("吴堡站水位多少", "水位 636"),
        ))
        assert len(sa.segment_rounds(rounds)) == 1

    def test_embedding_failure_single_segment(self, archive_env):
        with patch("agent.rag.embedding.embed_texts", return_value=None):
            rounds = sa.extract_rounds(_round_history(
                ("吴堡站水情如何", "a"), ("吕梁天气怎么样", "b"),
            ))
            segs = sa.segment_rounds(rounds)
            assert len(segs) == 1  # 降级单段

    def test_boundaries_only_grow(self, archive_env):
        """KV 不变量 3：追加新轮次不改旧边界。"""
        rounds = sa.extract_rounds(_round_history(
            ("吴堡站水情如何", "a"), ("吕梁天气怎么样", "b"),
        ))
        old = sa.segment_rounds(rounds)
        old_fps = [s.first_fp for s in old]
        rounds.append(sa.Round(query="吴堡流量预测", answer="c"))
        new = sa.segment_rounds(rounds)
        assert [s.first_fp for s in new][:len(old_fps)] == old_fps
        assert len(new) >= len(old)


# ============ 归档（幂等 + 工具数据） ============

class TestArchive:
    def test_archive_idempotent(self, archive_env):
        rounds = sa.extract_rounds(_round_history(
            ("吴堡站水情如何", "流量 537"), ("吴堡站水位多少", "水位 636"),
        ))
        segs = sa.segment_rounds(rounds)
        sa.archive_rounds(segs)
        files1 = sorted(p.name for p in archive_env.glob("*.md"))
        sa.archive_rounds(segs)  # 二次归档幂等
        files2 = sorted(p.name for p in archive_env.glob("*.md"))
        assert files1 == files2 and len(files1) == 1
        meta = sa._load_meta(segs[0].first_fp)
        assert meta["round_fps"] == [r.fp for r in rounds]

    def test_completed_round_with_tool_data(self, archive_env):
        history = _round_history(("吴堡站水情如何", "流量 537"))
        sa._archive_completed_round_sync(
            history, "吴堡站水位多少", "水位 636.14，低于警戒",
            [{"tool_name": "get_hydrology", "arguments": {"station": "吴堡"},
              "result": {"flow_m3_s": 582}, "error": ""}],
            warning_level="IV",
        )
        segs = sa.segment_rounds(sa.extract_rounds(history))
        body = sa._read_body(segs[0].first_fp)
        assert "get_hydrology" in body          # 工具轨迹已落盘
        assert "预警等级：IV" in body
        meta = sa._load_meta(segs[0].first_fp)
        assert len(meta["round_fps"]) == 2      # 本轮已追加进段

    def test_tool_data_overrides_plain(self, archive_env):
        """收尾归档的含工具版本覆盖入口归档的纯文本版本。"""
        r = sa.Round(query="吴堡站水情如何", answer="流量 537")
        seg = sa.Segment(rounds=[r])
        sa.upsert_round(seg.first_fp, seg.rounds, r, tool_data="")
        sa.upsert_round(seg.first_fp, seg.rounds, r, tool_data="- get_hydrology → ok")
        body = sa._read_body(seg.first_fp)
        assert "get_hydrology" in body
        meta = sa._load_meta(seg.first_fp)
        assert meta["round_fps"] == [r.fp]  # 轮数不重复


# ============ 冻结摘要（KV 不变量 1、2） ============

class TestFrozenSummaries:
    def test_summary_generated_once_and_frozen(self, archive_env):
        counter = {"n": 0}
        rounds = sa.extract_rounds(_round_history(
            ("吴堡站水情如何", "流量 537"), ("吴堡站水位多少", "水位 636"),
        ))
        seg = sa.Segment(rounds=rounds)
        sa.archive_rounds([seg])
        with patch("app.core.llm.get_llm_client", _mock_summary_llm(counter)):
            s1 = sa.ensure_summaries(seg, needed_covered=2, seg_index=0)
            s2 = sa.ensure_summaries(seg, needed_covered=2, seg_index=0)
        assert s1 == s2                    # 同一冻结文本
        assert counter["n"] == 1           # LLM 只调一次
        # 文件固化后不依赖 LLM 也能读到同一摘要
        s3 = sa.ensure_summaries(seg, needed_covered=2, seg_index=0)
        assert s3 == s1

    def test_continuation_summary_appends(self, archive_env):
        """段追加新轮：旧摘要不变，续摘要追加。"""
        counter = {"n": 0}
        r1 = sa.Round(query="吴堡站水情如何", answer="流量 537")
        seg2 = sa.Segment(rounds=[r1])
        sa.archive_rounds([seg2])
        with patch("app.core.llm.get_llm_client", _mock_summary_llm(counter)):
            first = sa.ensure_summaries(seg2, needed_covered=1, seg_index=0)

        r2 = sa.Round(query="吴堡站水位多少", answer="水位 636")
        seg4 = sa.Segment(rounds=[r1, r2])
        with patch("app.core.llm.get_llm_client", _mock_summary_llm(counter)):
            second = sa.ensure_summaries(seg4, needed_covered=2, seg_index=0)

        assert len(second) == 2                    # 初始 + 续
        assert second[0] == first[0]               # 旧摘要逐字不变（不变量 1）
        assert counter["n"] == 2                   # 只为新增轮调 LLM

    def test_llm_failure_fallback_not_frozen(self, archive_env):
        rounds = sa.extract_rounds(_round_history(("吴堡站水情如何", "流量 537")))
        seg = sa.Segment(rounds=rounds)
        sa.archive_rounds([seg])
        with patch("app.core.llm.get_llm_client", side_effect=Exception("llm down")):
            texts = sa.ensure_summaries(seg, needed_covered=1, seg_index=0)
        assert len(texts) == 1 and "意图" in texts[0]   # 规则提取降级
        assert sa._load_meta(seg.first_fp)["summaries"] == []  # 未冻结（下次重试）


# ============ 压缩产出 ============

class TestCompactWithSegments:
    def _long_history(self):
        """6 轮三主题（水情/天气/水情），每轮答案够长以超预算。"""
        return _round_history(
            ("吴堡站水情如何", "流量 537，水位 636.06，低于警戒水位 640 共 3.94 米。" * 20),
            ("吴堡站水位多少", "水位 636.06 米。" * 40),
            ("吕梁天气怎么样", "晴，气温 26 度。" * 40),
            ("明天降雨大吗", "小雨。" * 40),
            ("吴堡站流量趋势", "未来 24 小时平稳。" * 40),
            ("吴堡站水位趋势", "持续低于警戒。" * 40),
        )

    def test_compact_produces_summaries_and_window(self, archive_env):
        counter = {"n": 0}
        history = self._long_history()
        with patch("agent.memory.session_archive._archive_async"), \
             patch("app.core.llm.get_llm_client", _mock_summary_llm(counter)):
            out = sa.compact_with_segments(history, keep_recent_rounds=2)
        # 头部是段摘要 system 消息，尾部是近 2 轮原文（4 条消息）
        assert out[0]["role"] == "system" and "[历史任务" in out[0]["content"]
        assert len(out[-4:]) == 4
        assert out[-1] == history[-1]
        assert out[-4]["content"] == history[-4]["content"]  # 近 2 轮原文保留

    def test_compact_output_stable_across_calls(self, archive_env):
        """KV 核心：同 history 压缩两次输出逐字一致（冻结生效）。"""
        counter = {"n": 0}
        history = self._long_history()
        with patch("agent.memory.session_archive._archive_async"), \
             patch("app.core.llm.get_llm_client", _mock_summary_llm(counter)):
            out1 = sa.compact_with_segments(history, keep_recent_rounds=2)
            out2 = sa.compact_with_segments(history, keep_recent_rounds=2)
        assert out1 == out2

    def test_uncompacted_history_unchanged(self, tmp_path, monkeypatch):
        """未超预算零开销返回原 history（compact_history 入口行为不变）。"""
        from agent.graph.context_compact import compact_history

        history = _round_history(("你好", "你好！"), ("在吗", "在的"))
        assert compact_history(history) is history


# ============ 按需还原 ============

class TestRecall:
    def test_recall_hits_relevant_segment(self, archive_env):
        history = _round_history(
            ("吴堡站水情如何", "流量 537，水位 636.06"),
            ("吴堡站水位多少", "水位 636.06 米"),
            ("吕梁天气怎么样", "晴"),
            ("明天降雨大吗", "小雨"),
        )
        segs = sa.segment_rounds(sa.extract_rounds(history))
        sa.archive_rounds(segs)
        recalled = sa.recall_relevant_segments(
            "吴堡站流量多少", history, keep_recent_rounds=1,
        )
        assert "流量 537" in recalled       # 还原了水情段全文
        assert "相关历史任务还原" in recalled

    def test_recall_no_match_returns_empty(self, archive_env):
        history = _round_history(
            ("吴堡站水情如何", "流量 537"),
            ("吴堡站水位多少", "水位 636"),
        )
        sa.archive_rounds(sa.segment_rounds(sa.extract_rounds(history)))
        # 预案主题与水情段正交（余弦 0）→ 无命中
        assert sa.recall_relevant_segments(
            "生成防汛应急预案", history, keep_recent_rounds=0,
        ) == ""

    def test_recall_keyword_fallback(self, archive_env):
        history = _round_history(("吴堡站水情如何", "流量 537"))
        sa.archive_rounds(sa.segment_rounds(sa.extract_rounds(history)))
        with patch("agent.rag.embedding.embed_texts", return_value=None):
            sa._EMB_CACHE.clear()
            recalled = sa.recall_relevant_segments(
                "吴堡站水情", history, keep_recent_rounds=0,
            )
        assert "流量 537" in recalled and "关键词降级" in recalled


# ============ 压缩标记双兼容 ============

class TestCompactedDetection:
    def test_dual_marker(self):
        from agent.graph.context_compact import is_compacted_history

        assert is_compacted_history(
            [{"role": "system", "content": "[历史任务·1] 意图：查水情"}]
        )
        assert is_compacted_history(
            [{"role": "system", "content": "[历史对话摘要]\n今天聊了水情"}]
        )
        assert not is_compacted_history([{"role": "user", "content": "hi"}])
