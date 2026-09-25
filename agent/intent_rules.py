"""意图规则档案：预案/研判类识别规则的唯一配置源（外置化）。

背景（2026-09-15 外置化）：预案类正则原为 planner_guard 内的代码常量，
口语措辞（"提几条处置建议"）不在词表内导致完成度闸漏触发、请求被路由进
闲聊。对齐 thresholds.json 的外置化模式收敛为本档案：

- config/intent_rules.json：唯一规则来源（version + 词表）。环境变量
  WATERAGENTS_INTENT_RULES_FILE 可覆盖路径；
- get_intent_rules()：进程内缓存的规则视图，返回编译好的正则；
- reload_intent_rules()：换配置后清缓存，下次读取即生效。

修订流程：改配置（version 递增）→ 单测（test_intent_rules /
test_planner_guard）→ 62 条评估重放过门禁。注意：提示词层的意图原则
（planner system prompt 规则 9）是主路径，本配置只兜漏网——新增业务
意图类型的完整动作是"配置加词表 + 提示词补原则"，两处都不碰 Python 代码。
"""
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "intent_rules.json"
ENV_CONFIG_PATH = "WATERAGENTS_INTENT_RULES_FILE"

# 配置缺失/损坏时的兜底（与 v2026.09 内容一致）——保证守卫永不失联
_FALLBACK = {
    "version": "fallback",
    "plan_request": {
        "verbs": ["生成", "制定", "编制", "输出", "安排", "提", "给出", "列出", "列", "推荐"],
        "nouns": ["应急预案", "应急响应方案", "处置预案", "处置方案", "应急方案",
                  "转移方案", "预案", "处置建议", "应对措施", "措施建议",
                  "建议措施", "处置措施"],
        "concept_suppress": ["组成部分", "包括哪些", "有哪些", "是什么", "什么是",
                             "含义", "定义", "区别", "分类", "种类", "判别标准"],
    },
    "assess_request": {
        "patterns": ["研判", "防汛形势", "洪水风险", "防汛压力",
                     "风险评[估判]", "综合评[估判]"],
    },
}

# 动词与名词之间的允许间隔（与原代码常量一致：同句内跨 ≤15 个字符）
_PLAN_GAP = r"[^。？！?!\n]{0,15}"


@dataclass(frozen=True)
class IntentRules:
    """一份意图规则（编译好的不可变快照）。

    concept_re：概念类问法抑制词（可空）。命中时预案闸不触发——
    "列出应急预案的组成部分"是概念解释，不是要一份预案。
    """

    version: str
    plan_request_re: re.Pattern
    assess_re: re.Pattern
    concept_re: re.Pattern | None = None


def _load_config() -> dict:
    path = Path(os.environ.get(ENV_CONFIG_PATH, DEFAULT_CONFIG_PATH))
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        _ = cfg["plan_request"]["verbs"]  # 结构粗校验
        return cfg
    except Exception as e:  # noqa: BLE001 - 配置问题降级为兜底，守卫不失联
        logger.warning("[intent_rules] 配置加载失败（%s），使用内置兜底词表：%s", path, e)
        return _FALLBACK


def _compile(cfg: dict) -> IntentRules:
    plan = cfg["plan_request"]
    verbs = "|".join(plan["verbs"])
    nouns = "|".join(plan["nouns"])
    plan_re = re.compile(rf"({verbs}){_PLAN_GAP}({nouns})")
    assess_re = re.compile("|".join(cfg["assess_request"]["patterns"]))
    suppress = plan.get("concept_suppress") or []
    concept_re = re.compile("|".join(suppress)) if suppress else None
    return IntentRules(version=str(cfg.get("version", "unknown")),
                       plan_request_re=plan_re, assess_re=assess_re,
                       concept_re=concept_re)


@lru_cache(maxsize=1)
def get_intent_rules() -> IntentRules:
    return _compile(_load_config())


def reload_intent_rules() -> IntentRules:
    """换配置后原地重载（测试与热更新用）。"""
    get_intent_rules.cache_clear()
    return get_intent_rules()
