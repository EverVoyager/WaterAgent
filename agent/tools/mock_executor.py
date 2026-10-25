"""工具分发与评估回放：真实源优先，按环境控制模拟降级。"""
import random
import threading
from contextlib import contextmanager
from typing import Any

from agent.tools.mock_results import _MOCK_IMPLEMENTATIONS
from agent.tools.schemas import (
    TOOL_PARAM_MODELS,
)

# ====== 执行器入口 ======

# 评估回放上下文（evals/ 使用）：激活后所有工具强制走 mock 分支并按
# case 注入 overrides+seed，保证"评估环境状态可重置、结果可复现"。
# 用模块级全局而非 contextvars：graph 内 ThreadPoolExecutor 不传播上下文变量；
# 评估按 case 顺序执行无竞争，锁仅防多线程评估器误用。
_REPLAY_LOCK = threading.Lock()

_REPLAY_ACTIVE = False

_REPLAY_OVERRIDES: dict[str, dict[str, Any]] = {}

_REPLAY_SEED: int | None = None


def set_replay_context(
    overrides_by_tool: dict[str, dict[str, Any]] | None = None,
    seed: int | None = None,
) -> None:
    """激活回放上下文：overrides_by_tool 为 工具名->覆盖值 映射，seed 为全局随机种子。"""
    global _REPLAY_ACTIVE, _REPLAY_OVERRIDES, _REPLAY_SEED
    with _REPLAY_LOCK:
        _REPLAY_ACTIVE = True
        _REPLAY_OVERRIDES = dict(overrides_by_tool or {})
        _REPLAY_SEED = seed
    # 回放环境切换必须清工具结果缓存，否则前一个 case 的结果会经 TTL 缓存串入
    from agent.graph.cache import clear_tool_result_cache
    clear_tool_result_cache()


def clear_replay_context() -> None:
    """退出回放上下文，恢复正常执行路径（真实实现优先）。"""
    global _REPLAY_ACTIVE, _REPLAY_OVERRIDES, _REPLAY_SEED
    with _REPLAY_LOCK:
        _REPLAY_ACTIVE = False
        _REPLAY_OVERRIDES = {}
        _REPLAY_SEED = None


def is_replay_active() -> bool:
    with _REPLAY_LOCK:
        return _REPLAY_ACTIVE


@contextmanager
def replay_context(
    overrides_by_tool: dict[str, dict[str, Any]] | None = None,
    seed: int | None = None,
):
    """回放上下文管理器：with 内工具走确定性 mock 回放，退出自动恢复。"""
    set_replay_context(overrides_by_tool, seed)
    try:
        yield
    finally:
        clear_replay_context()

# 透传工具（无外部依赖、读磁盘/内存即得）：直接走 real_executor，不降级 mock
# - list_skills：技能清单元工具（对标 MCP tools/list）
# - read_session_archive：历史任务段还原（读磁盘归档，mock 无法模拟用户历史）
_PASSTHROUGH_TOOLS = {"list_skills", "read_session_archive", "read_memory_topic"}

# 关键业务工具：生产环境真实实现不可用时硬失败，不降级到 mock。
# 防汛决策不能基于模拟数据——返回 mock 水情/法规/预案会误导指挥决策。
_CRITICAL_TOOLS = set(_MOCK_IMPLEMENTATIONS.keys())


def _should_hard_fail_on_real_failure(tool_name: str) -> bool:
    """生产环境下关键业务工具真实实现不可用时应硬失败。

    development 环境：始终返回 False（降级到 mock，方便本地无依赖开发）
    production 环境 + 关键工具：返回 True（raise，错误直传前端）
    """
    from app.core.config import get_settings
    return get_settings().is_production and tool_name in _CRITICAL_TOOLS


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """统一工具执行入口。

    优先调用 real_executor 中的真实实现（如 RAG 法规检索），
    未覆盖或索引未就绪时按环境分级处理：
    - development：降级到 mock 实现
    - production：关键业务工具硬失败（raise），错误直传前端

    Args:
        tool_name: 工具名称
        arguments: 工具入参 dict
        overrides: 训练侧确定性回放用，仅作用于 mock 分支；真实实现分支忽略
        seed: 训练侧确定性回放用，设全局 random 种子；真实实现分支忽略

    Returns:
        工具执行结果 dict

    Raises:
        ValueError: 工具名未知或入参不合法
        RuntimeError: production 环境关键工具真实实现不可用（依赖未就绪等）
    """
    # 透传工具（如 list_skills）无 mock 实现，直接走 real_executor
    if tool_name in _PASSTHROUGH_TOOLS:
        from agent.tools.real_executor import real_execute_tool
        return real_execute_tool(tool_name, arguments)

    if tool_name not in _MOCK_IMPLEMENTATIONS:
        raise ValueError(f"Unknown tool: {tool_name}")

    # 评估回放上下文激活时：强制走 mock 分支（真实实现返回非确定数据会破坏
    # "可重置可复现"的评估环境约定），overrides/seed 取自回放上下文。
    # 显式传入的 overrides/seed（训练侧用法）优先级更高，不受影响。
    if overrides is None and seed is None:
        with _REPLAY_LOCK:
            replay_on = _REPLAY_ACTIVE
            ctx_overrides = _REPLAY_OVERRIDES.get(tool_name)
            ctx_seed = _REPLAY_SEED
        if replay_on:
            overrides, seed = ctx_overrides, ctx_seed
    else:
        replay_on = False

    # 训练侧确定性：设全局种子（仅 mock 分支依赖 random）
    if seed is not None:
        random.seed(seed)

    # overrides/seed 是训练侧确定性回放信号：跳过 real 分支，直接走 mock，
    # 避免真实实现（如 qqjjsj.com 爬虫）返回非确定数据覆盖 overrides。
    # 正常运行（overrides=seed=None）仍优先 real_executor。
    if overrides is None and seed is None and not replay_on:
        # 优先尝试真实实现（阶段 D 起：search_regulation 走 RAG）
        try:
            from agent.tools.real_executor import real_execute_tool
            result = real_execute_tool(tool_name, arguments)
            if not result.get("__not_implemented__"):
                return result
        except RuntimeError as e:
            # 真实实现依赖未就绪（如 FAISS 索引未构建）
            if _should_hard_fail_on_real_failure(tool_name):
                raise
            import logging
            logging.getLogger(__name__).warning(
                "[executor] %s 真实实现不可用，降级到 mock: %s", tool_name, e,
            )
        except Exception as e:
            if _should_hard_fail_on_real_failure(tool_name):
                raise
            import logging
            logging.getLogger(__name__).exception(
                "[executor] %s 真实实现异常，降级到 mock: %s", tool_name, e,
            )

    # 回退到 mock
    param_model = TOOL_PARAM_MODELS[tool_name]
    # Pydantic 校验入参
    try:
        params = param_model(**arguments)
    except Exception as e:
        raise ValueError(f"Invalid arguments for {tool_name}: {e}") from e

    impl = _MOCK_IMPLEMENTATIONS[tool_name]
    mock_result = impl(params, overrides=overrides)
    # 标记数据来源，便于调试
    if isinstance(mock_result, dict):
        mock_result["source"] = "mock"
    return mock_result
