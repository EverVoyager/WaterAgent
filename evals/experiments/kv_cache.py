"""KV Cache 前缀冻结实验：冻结 vs 破坏前缀的命中率对照（vLLM 式工程口径）。

设计（对照而非消融——"冻结"是默认生产行为，反面是"破坏"）：
- 冻结版：生产行为原样（静态前缀：分层系统提示 + 排序工具 schema）；
- 破坏版：planner 系统提示尾部追加每次调用都变的 nonce（时间戳），
  前缀逐请求漂移 → 前缀缓存无法命中。synthesizer 等节点不动，
  因此对照聚焦 planner 节点、total 为混合口径（报告按节点呈现）。

测量：llm_stats 进程内按节点聚合 cached_tokens/prompt_tokens
（三后端字段兼容提取已在 llm_stats 内处理），会话脚本固定为
3 轮同站研判（回放环境 mock 工具，确定性）。

对外声明形如："多轮会话 planner 节点前缀命中率 X% → Y%（破坏 → 冻结），
等效 token 开销 −Z%"。
"""
import logging
import random
import time
from unittest.mock import patch

from evals.cases import EvalCase
from evals.replay import case_env
from evals.runner import run_graph_agent
from train.data_gen.scenario import _make_overrides

logger = logging.getLogger(__name__)

_PLANNER_PROMPT_TARGET = "agent.graph.nodes._build_planner_system_prompt"

_SESSION_QUERY_TPL = [
    "查一下{station}水文站现在的实时水情。",
    "结合雨水情，研判一下{station}站未来24小时的防汛形势。",
    "给{station}站当前的形势提几条处置建议。",
]

_SCRIPT_SEED = 399_000  # 会话脚本的 overrides 抽取种子（评估区间内、固定）


def _run_session_script(station: str, model_label: str = "") -> None:
    """同站 3 轮会话（回放环境）：第 2/3 轮与前轮共享长前缀，可观测命中。"""
    case = EvalCase(
        case_id="kv-script",
        case_type="business",
        query="",
        seed=_SCRIPT_SEED,
        overrides=_make_overrides(random.Random(_SCRIPT_SEED), station, "II"),
    )
    history: list[dict] = []
    with case_env(case):
        for tpl in _SESSION_QUERY_TPL:
            query = tpl.format(station=station)
            result = run_graph_agent(query, history=list(history))
            history.append({"role": "user", "content": query})
            history.append({
                "role": "assistant",
                "content": result.get("final_answer", ""),
            })


def _stats_snapshot() -> dict:
    """llm_stats 聚合快照 → 每节点 + 总计命中率。"""
    from app.core.llm_stats import get_cache_stats

    stats = get_cache_stats()
    nodes = {
        node: {
            "calls": s["calls"],
            "prompt_tokens": s["prompt_tokens"],
            "cached_tokens": s["cached_tokens"],
            "hit_rate": (
                round(s["cached_tokens"] / s["prompt_tokens"], 4)
                if s["prompt_tokens"] else 0.0
            ),
        }
        for node, s in stats.items()
    }
    prompt = sum(s["prompt_tokens"] for s in stats.values())
    cached = sum(s["cached_tokens"] for s in stats.values())
    return {
        "nodes": nodes,
        "total": {
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "hit_rate": round(cached / prompt, 4) if prompt else 0.0,
        },
    }


def run_kv_cache_experiment(station: str = "吴堡", model_label: str = "") -> dict:
    """前缀冻结 vs 破坏：同会话脚本各跑一遍，比对 llm_stats 命中率。"""
    from app.core.llm_stats import reset_cache_stats

    import agent.graph.nodes as nodes_mod

    # 冻结版：生产行为（统计隔离：每版跑前清零）
    reset_cache_stats()
    _run_session_script(station, model_label=model_label)
    frozen = _stats_snapshot()

    # 破坏版：planner 系统提示尾部追加逐调用变化的 nonce
    reset_cache_stats()
    original_prompt = nodes_mod._build_planner_system_prompt

    def _broken_prompt() -> str:
        return original_prompt() + f"\n<!-- eval-nonce:{time.time_ns()} -->"

    with patch(_PLANNER_PROMPT_TARGET, new=_broken_prompt):
        _run_session_script(station, model_label=model_label)
    broken = _stats_snapshot()

    frozen_planner = frozen["nodes"].get("planner", {}).get("hit_rate")
    broken_planner = broken["nodes"].get("planner", {}).get("hit_rate")
    planner_delta = (
        round(frozen_planner - broken_planner, 4)
        if frozen_planner is not None and broken_planner is not None else None
    )

    return {
        "experiment": "kv_cache",
        "frozen": frozen,
        "broken": broken,
        "planner_hit_frozen": frozen_planner,
        "planner_hit_broken": broken_planner,
        "planner_hit_delta": planner_delta,
        "note": (
            "对照聚焦 planner 节点（仅破坏其前缀）；total 为混合口径。"
            "命中数字来自 llm_stats 进程内观测，要求推理后端返回"
            "cached_tokens 字段（OpenAI 风格 / DeepSeek / vLLM 均兼容）。"
        ),
    }
