"""意图路由模块。

参考 aurelio-labs/semantic-router 的设计思路，自研轻量版：
- SemanticRoute: 单个意图（含名称 + 示例 utterances）
- SemanticRouter: 余弦相似度匹配，返回最接近的意图
"""
from agent.router.intent import detect_intent
from agent.router.semantic_router import SemanticRoute, SemanticRouter

__all__ = ["SemanticRoute", "SemanticRouter", "detect_intent"]
