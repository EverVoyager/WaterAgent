"""llm_stats 持久化与离线聚合报表测试。

覆盖：JSONL 持久化（含环境变量开关与失败防御）、summarize_records 的命中率/节省
估算口径、CLI 聚合入口 aggregate_file。
"""
import json

from app.core.llm_stats import (
    aggregate_file,
    format_summary,
    record_llm_usage,
    reset_cache_stats,
    summarize_records,
)


class TestPersistence:
    def test_records_written_when_env_set(self, tmp_path, monkeypatch):
        stats_file = tmp_path / "stats.jsonl"
        monkeypatch.setenv("LLM_STATS_FILE", str(stats_file))
        reset_cache_stats()
        record_llm_usage("planner", {"prompt_tokens": 1000,
                                     "prompt_tokens_details": {"cached_tokens": 600}})
        record_llm_usage("chat", {"prompt_tokens": 200})
        lines = stats_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["node"] == "planner"
        assert first["prompt_tokens"] == 1000
        assert first["cached_tokens"] == 600

    def test_no_file_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LLM_STATS_FILE", raising=False)
        # 先在 env 开启时写一条，确认关闭后不再追加
        stats_file = tmp_path / "stats.jsonl"
        monkeypatch.setenv("LLM_STATS_FILE", str(stats_file))
        record_llm_usage("chat", {"prompt_tokens": 100})
        monkeypatch.delenv("LLM_STATS_FILE", raising=False)
        record_llm_usage("chat", {"prompt_tokens": 100})
        assert len(stats_file.read_text().splitlines()) == 1

    def test_persistence_failure_never_raises(self, monkeypatch):
        # 指向一个不可能写入的路径：主流程必须无感
        monkeypatch.setenv("LLM_STATS_FILE", "Z://\\nonexistent/dir/stats.jsonl")
        record_llm_usage("chat", {"prompt_tokens": 100})


class TestSummarizeRecords:
    def test_hit_rate_and_savings(self):
        records = [
            {"node": "planner", "prompt_tokens": 1000, "cached_tokens": 600},
            {"node": "planner", "prompt_tokens": 1000, "cached_tokens": 600},
            {"node": "chat", "prompt_tokens": 200, "cached_tokens": 0},
        ]
        summary = summarize_records(records)
        planner = summary["nodes"]["planner"]
        assert planner["calls"] == 2
        assert planner["hit_rate"] == 0.6
        # 命中 1200 token 按 0.25 计费：2000 - 1200*0.75 = 1100
        assert planner["cost_discounted"] == 1100.0
        total = summary["total"]
        assert total["prompt_tokens"] == 2200
        assert total["cached_tokens"] == 1200
        assert total["hit_rate"] == round(1200 / 2200, 4)

    def test_cached_clamped_to_prompt(self):
        """后端异常多报 cached > prompt 时钳制，不产生负开销"""
        summary = summarize_records(
            [{"node": "x", "prompt_tokens": 100, "cached_tokens": 300}]
        )
        assert summary["total"]["cost_discounted"] >= 0
        assert summary["total"]["cached_tokens"] == 100

    def test_empty_records(self):
        summary = summarize_records([])
        assert summary["total"]["prompt_tokens"] == 0
        assert summary["total"]["saved_pct"] == 0.0


class TestAggregateFile:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "stats.jsonl"
        path.write_text(
            json.dumps({"node": "planner", "prompt_tokens": 1000, "cached_tokens": 500})
            + "\n" + "损坏行\n\n",  # 容错：跳过不可解析行与空行
            encoding="utf-8",
        )
        summary = aggregate_file(path)
        assert summary["nodes"]["planner"]["hit_rate"] == 0.5

    def test_format_summary_renders(self):
        summary = summarize_records(
            [{"node": "synthesizer_phase2", "prompt_tokens": 800, "cached_tokens": 640}]
        )
        text = format_summary(summary)
        assert "synthesizer_phase2" in text
        assert "TOTAL" in text
        assert "80.0%" in text  # 命中率
