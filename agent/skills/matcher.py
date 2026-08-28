"""Skill 匹配引擎（借鉴 Claude Skills 按需加载机制）。

工作流程：
1. 启动时预计算所有 enabled skills 的 description embedding（缓存）
2. 用户提问时计算 query embedding，与所有 skill description 做余弦相似度
3. 取最高分，超过阈值则匹配，加载完整 instructions
4. embedding 不可用时降级到关键词匹配

与 Claude Skills 一致的设计：
- 只扫描 description（轻量），匹配后才加载完整 instructions（按需）
- 多个 skill 匹配时取最高分（不做组合，保持简单）
"""
import logging
import threading

import numpy as np

from agent.rag.embedding import embed_query, embed_texts
from agent.skills.models import Skill
from agent.skills.store import list_skills

logger = logging.getLogger(__name__)

# 匹配阈值：低于此分数不匹配（与 semantic_router 一致）
_MATCH_THRESHOLD = 0.55

# 关键词匹配时需剥离的首尾标点字符集（strip 按字符集语义使用）
_PUNCT_CHARS = "，。、；：！？\"\"''（）()【】[]《》<>"


class _SkillMatcher:
    """Skill 匹配器（单例，带缓存）。

    缓存策略：
    - 预计算所有 enabled skills 的 description embedding
    - skill 列表变更时（create/update/delete）调 invalidate() 失效缓存
    - 下次匹配时自动重建
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._skills: list[Skill] = []
        self._embeddings: np.ndarray | None = None  # (N, 1024)
        self._dirty = True  # 是否需要重建缓存

    def invalidate(self) -> None:
        """失效缓存（skill 增删改后调用）。"""
        with self._lock:
            self._dirty = True
            self._skills = []
            self._embeddings = None

    def _ensure_cache(self) -> None:
        """确保缓存就绪（脏时重建）。"""
        if not self._dirty and self._embeddings is not None:
            return

        with self._lock:
            if not self._dirty and self._embeddings is not None:
                return

            self._skills = list_skills(enabled_only=True)
            if not self._skills:
                self._embeddings = np.zeros((0, 1024), dtype=np.float32)
                self._dirty = False
                logger.info("[skill-matcher] 无 enabled skills")
                return

            descriptions = [s.description for s in self._skills]
            try:
                self._embeddings = embed_texts(descriptions)
                if self._embeddings is None or self._embeddings.shape[0] == 0:
                    raise RuntimeError("embed_texts 返回空")
                logger.info(
                    "[skill-matcher] 缓存就绪: %d skills, shape=%s",
                    len(self._skills), self._embeddings.shape,
                )
            except Exception as e:
                logger.warning("[skill-matcher] embedding 预计算失败: %s — 将降级到关键词匹配", e)
                self._embeddings = None

            self._dirty = False

    def match(self, query: str) -> tuple[Skill | None, float]:
        """匹配 query 到最相关的 Skill。

        Returns:
            (skill, score)：未匹配时 skill=None, score=0.0
        """
        self._ensure_cache()

        if not self._skills:
            return None, 0.0

        # 优先用 embedding 语义匹配
        if self._embeddings is not None and self._embeddings.shape[0] > 0:
            try:
                q_emb = embed_query(query)
                if q_emb is not None:
                    # 内积 = 余弦相似度（向量已 L2 归一化）
                    scores = self._embeddings @ q_emb
                    best_idx = int(np.argmax(scores))
                    best_score = float(scores[best_idx])
                    if best_score >= _MATCH_THRESHOLD:
                        logger.info(
                            "[skill-matcher] embedding 匹配: skill=%s score=%.3f",
                            self._skills[best_idx].name, best_score,
                        )
                        return self._skills[best_idx], best_score
                    logger.debug(
                        "[skill-matcher] embedding 最高分 %.3f < 阈值 %.2f，未匹配",
                        best_score, _MATCH_THRESHOLD,
                    )
                    return None, best_score
            except Exception as e:
                logger.warning("[skill-matcher] query embedding 失败: %s — 降级到关键词", e)

        # 降级：关键词匹配
        return self._match_by_keywords(query)

    def _match_by_keywords(self, query: str) -> tuple[Skill | None, float]:
        """关键词匹配（embedding 不可用时的降级方案）。

        策略：把 description 分词，计算 query 命中的关键词比例作为分数。
        """
        query_lower = query.lower()
        best_skill = None
        best_score = 0.0

        for skill in self._skills:
            # 简单分词：按空格/标点切分，过滤短词
            words = [
                w.lower().strip(_PUNCT_CHARS)
                for w in skill.description.replace(",", " ").split()
            ]
            words = [w for w in words if len(w) >= 2]
            if not words:
                continue
            hits = sum(1 for w in words if w in query_lower)
            score = hits / len(words)
            if score > best_score:
                best_score = score
                best_skill = skill

        # 关键词匹配阈值放宽（命中 30% 即匹配）
        if best_skill and best_score >= 0.3:
            logger.info(
                "[skill-matcher] 关键词匹配: skill=%s score=%.3f",
                best_skill.name, best_score,
            )
            return best_skill, best_score

        return None, best_score


# 单例
_matcher = _SkillMatcher()


def match_skill(query: str) -> Skill | None:
    """匹配 query 到最相关的 Skill。未匹配返回 None。"""
    skill, _ = _matcher.match(query)
    return skill


def get_active_skill_instructions(query: str) -> str | None:
    """获取匹配到的 Skill 的完整指令（按需加载）。

    未匹配到 Skill 时返回 None（调用方用默认 prompt）。
    """
    skill = match_skill(query)
    if skill is None:
        return None
    logger.info("[skill-matcher] 加载技能指令: %s", skill.name)
    return skill.instructions


def invalidate_cache() -> None:
    """失效匹配缓存（skill 增删改后调用）。"""
    _matcher.invalidate()
