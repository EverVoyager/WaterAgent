"""KV Cache 前缀冻结实验：冻结 vs 破坏前缀的命中率对照（vLLM 式工程口径）。

设计（对照而非消融——"冻结"是默认生产行为，反面是"破坏"）：
- 冻结版：生产行为原样（静态前缀：分层系统提示 + 排序工具 schema）；
- 破坏版：planner 系统提示尾部追加每次调用都变的 nonce（时间戳），
  前缀逐请求漂移 → 前缀缓存无法命中。synthesizer 等节点不动，
  因此对照聚焦 planner 节点、total 为混合口径（报告按节点呈现）。

测量：llm_stats 进程内按节点聚合 cached_tokens/prompt_tokens
（三后端字段兼容提取已在 llm_stats 内处理），会话脚本固定为
6 轮同站多意图（水情→研判→预案→趋势→处置措施→退水研判，
回放环境 mock 工具，确定性）。轮数代表真实多轮会话的稳态负载——
共享前缀占比随轮数上升，报告口径时须注明会话长度。

稳态口径（2026-09-15 起）：每臂计量前先空跑一遍会话预热。生产环境
缓存常热，冷启动首调的 miss 不代表稳态表现；且两臂先后顺序会因账号级
隐式缓存的预热产生偏置（预热均衡后先后无差别）。环境变量
KV_CACHE_WARMUP=0 可关闭预热（冷口径，默认开）。

对外声明形如："多轮会话 planner 节点前缀命中率 X% → Y%（破坏 → 冻结），
等效 token 开销 −Z%"。
"""
import logging
import os
import random
import time
from unittest.mock import patch

from evals.cases import EvalCase
from evals.experiments.prompt_capture import capture_prompts
from evals.replay import case_env
from evals.runner import run_graph_agent
from train.data_gen.scenario import _make_overrides

logger = logging.getLogger(__name__)

_PLANNER_PROMPT_TARGET = "agent.graph.nodes._build_planner_system_prompt"

_SESSION_QUERY_TPL = [
    "查一下{station}水文站现在的实时水情。",
    "结合雨水情，研判一下{station}站未来24小时的防汛形势。",
    "给{station}站当前的形势提几条处置建议。",
    "明天{station}站上游有强降雨，帮忙研判一下洪峰量级和趋势。",
    "根据目前的水情，按流程给我列一份{station}站的应急处置措施。",
    "综合前面的情况，研判{station}站退水阶段还有哪些风险。",
]

_SCRIPT_SEED = 399_000  # 会话脚本的 overrides 抽取种子（评估区间内、固定）


def _run_session_script(station: str, model_label: str = "") -> None:
    """同站 6 轮多意图会话（回放环境）：后续轮次与前轮共享长前缀，可观测命中。"""
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
    """前缀冻结 vs 破坏：同会话脚本各跑一遍，比对 llm_stats 命中率。

    稳态口径：每臂计量前空跑一遍预热（KV_CACHE_WARMUP=0 关闭）。
    设置环境变量 ``KV_PROMPT_DUMP=<dir>`` 时，两臂的请求 prompt 分别
    落盘到 ``<dir>/frozen.jsonl`` / ``<dir>/broken.jsonl``，供
    ``evals.experiments.prefix_reuse_ratio`` 离线计算静态前缀复用占比
    （预热不落盘——落盘只含计量轮）。
    """
    import agent.graph.nodes as nodes_mod
    from app.core.llm_stats import reset_cache_stats

    dump_dir = os.environ.get("KV_PROMPT_DUMP", "").strip() or None
    warm = os.environ.get("KV_CACHE_WARMUP", "1") != "0"

    def _measured(arm_file: str) -> dict:
        with capture_prompts(
            os.path.join(dump_dir, arm_file) if dump_dir else None
        ):
            _run_session_script(station, model_label=model_label)
        return _stats_snapshot()

    # ===== 冻结臂（生产行为）=====
    reset_cache_stats()
    if warm:
        _run_session_script(station, model_label=model_label)  # 预热，不计入
    reset_cache_stats()
    frozen = _measured("frozen.jsonl")

    # ===== 破坏臂：planner 系统提示尾部追加逐调用变化的 nonce =====
    reset_cache_stats()
    original_prompt = nodes_mod._build_planner_system_prompt

    def _broken_prompt() -> str:
        return original_prompt() + f"\n<!-- eval-nonce:{time.time_ns()} -->"

    with patch(_PLANNER_PROMPT_TARGET, new=_broken_prompt):
        if warm:
            _run_session_script(station, model_label=model_label)  # 预热，不计入
        reset_cache_stats()
        broken = _measured("broken.jsonl")

    frozen_planner = frozen["nodes"].get("planner", {}).get("hit_rate")
    broken_planner = broken["nodes"].get("planner", {}).get("hit_rate")
    planner_delta = (
        round(frozen_planner - broken_planner, 4)
        if frozen_planner is not None and broken_planner is not None else None
    )

    result = {
        "experiment": "kv_cache",
        "warmup": warm,
        "session_rounds": len(_SESSION_QUERY_TPL),
        "frozen": frozen,
        "broken": broken,
        "planner_hit_frozen": frozen_planner,
        "planner_hit_broken": broken_planner,
        "planner_hit_delta": planner_delta,
        "note": (
            "对照聚焦 planner 节点（仅破坏其前缀）；total 为混合口径。"
            "命中数字来自 llm_stats 进程内观测，要求推理后端返回"
            "cached_tokens 字段（OpenAI 风格 / DeepSeek / vLLM 均兼容）。"
            "稳态口径：每臂计量前空跑一遍预热（KV_CACHE_WARMUP=0 关闭）。"
        ),
    }
    if dump_dir:
        result["prompt_dump_dir"] = dump_dir
    return result
