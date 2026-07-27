"""意图识别入口：Semantic Router 主路径 + 规则化兜底。

调用顺序：
    1. Semantic Router（embedding 余弦相似度，置信度 ≥ 阈值）
    2. 规则化兜底（关键词 + 长度判断）

M7：阈值从 config.SEMANTIC_ROUTER_THRESHOLD 读取，可热更新。
"""
import logging
from functools import lru_cache

from agent.router.routes import ROUTES, ROUTE_NEEDS_AGENT
from agent.router.semantic_router import RouteDecision, SemanticRouter
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# 防汛业务关键词（规则化兜底用）
_AGENT_KEYWORDS = (
    "水情", "水位", "流量", "洪峰", "径流", "降雨", "暴雨", "天气", "气象",
    "预警", "汛情", "防汛", "防洪", "险情", "巡堤", "查险", "险工",
    "吴堡", "龙门", "府谷", "吕梁", "黄河",
    "应急", "响应", "预案", "转移", "安置", "物资", "调度",
    "法规", "条例", "办法", "规定", "标准",
    "Ⅰ级", "Ⅱ级", "Ⅲ级", "Ⅳ级", "一级", "二级", "三级", "四级",
    "红色", "橙色", "黄色", "蓝色",
    "GIS", "地形", "河床", "淹没", "DEM",
)

# 短问候词（极短输入兜底）
_GREETING_KEYWORDS = ("你好", "您好", "hi", "hello", "嗨", "在吗", "在", "哈喽")


@lru_cache(maxsize=1)
def _get_semantic_router() -> SemanticRouter:
    """单例 SemanticRouter，启动时预计算 embedding。

    M7：阈值从 config 读取，便于调优而无需改代码。
    """
    settings = get_settings()
    threshold = settings.SEMANTIC_ROUTER_THRESHOLD
    logger.info("[intent] semantic_router threshold=%.3f", threshold)
    return SemanticRouter(routes=ROUTES, threshold=threshold)


def _rule_based_fallback(query: str) -> str:
    """规则化兜底意图识别。"""
    q = query.strip().lower()
    if not q:
        return "chitchat"
    # 极短输入 + 问候词
    if len(q) <= 6 and any(k in q for k in _GREETING_KEYWORDS):
        return "chitchat"
    # 含业务关键词
    if any(k in query for k in _AGENT_KEYWORDS):
        return "agent_task"
    # 默认闲聊（避免误判）
    return "chitchat"


def detect_intent(query: str) -> tuple[str, RouteDecision]:
    """识别用户意图。

    Args:
        query: 用户输入

    Returns:
        (intent, decision) 元组
        - intent: "chitchat" | "agent_task"
        - decision: RouteDecision，含匹配详情
    """
    router = _get_semantic_router()

    # 主路径：semantic router
    if router.ready:
        decision = router(query)
        if decision.route_name:
            logger.info(
                "[intent] semantic hit: %s (score=%.3f, utt=%s)",
                decision.route_name, decision.score, decision.matched_utterance[:30],
            )
            return decision.route_name, decision

        # 未命中阈值，记录降级原因
        logger.info("[intent] semantic miss, fallback: %s", decision.fallback_reason)

    # 兜底：规则化
    intent = _rule_based_fallback(query)
    decision = RouteDecision(
        route_name=intent,
        score=0.0,
        fallback_reason="rule_based_fallback",
    )
    logger.info("[intent] rule-based: %s", intent)
    return intent, decision


def intent_needs_agent(intent: str) -> bool:
    """判断意图是否需要进入 Agent 流程。"""
    return ROUTE_NEEDS_AGENT.get(intent, False)
