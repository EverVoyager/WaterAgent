"""意图规则配置（config/intent_rules.json）加载与匹配测试。

背景：口语预案请求（"提几条处置建议"）曾因词表缺词漏过完成度闸、被路由
进闲聊（2026-09-15 kv-cache dump diff 定位）。词表外置后，加词=改配置。
"""
import json

from agent.intent_rules import get_intent_rules, reload_intent_rules


def test_version_loaded_from_config():
    assert reload_intent_rules().version == "v2026.09"


def test_colloquial_plan_request_matches():
    """口语措辞必须命中（回归锁：biz 路由进闲聊的漏网场景）。"""
    plan_re = get_intent_rules().plan_request_re
    for q in (
        "给吴堡站当前的形势提几条处置建议。",
        "给我推荐几个应对措施。",
        "帮我列出处置措施。",
    ):
        assert plan_re.search(q), q


def test_standard_plan_request_still_matches():
    assert get_intent_rules().plan_request_re.search("请生成应急处置预案。")
    assert get_intent_rules().plan_request_re.search("请为府谷站制定Ⅳ级预警下的转移方案。")


def test_concept_and_chitchat_not_matched():
    plan_re = get_intent_rules().plan_request_re
    for q in (
        "处置预案一般包括哪些内容？",  # 概念解释，不触发预案闸
        "你好，你是谁？",
        "四级预警分别是什么含义？",
    ):
        assert not plan_re.search(q), q


def test_assess_patterns_still_work():
    rules = get_intent_rules()
    assert rules.assess_re.search("帮我研判一下防汛形势")
    assert not rules.assess_re.search("你好，你是谁？")


def test_env_override_and_reload(tmp_path, monkeypatch):
    cfg = {
        "version": "vTEST",
        "plan_request": {"verbs": ["生成"], "nouns": ["预案"]},
        "assess_request": {"patterns": ["研判"]},
    }
    p = tmp_path / "intent_rules.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("WATERAGENTS_INTENT_RULES_FILE", str(p))
    rules = reload_intent_rules()
    assert rules.version == "vTEST"
    assert rules.plan_request_re.search("生成预案")
    assert not rules.plan_request_re.search("提几条处置建议")  # 词表外不匹配
    monkeypatch.delenv("WATERAGENTS_INTENT_RULES_FILE")
    reload_intent_rules()  # 恢复默认配置视图


def test_fallback_on_broken_config(tmp_path, monkeypatch):
    p = tmp_path / "broken.json"
    p.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("WATERAGENTS_INTENT_RULES_FILE", str(p))
    rules = reload_intent_rules()
    assert rules.version == "fallback"
    assert rules.plan_request_re.search("请生成应急处置预案。")  # 守卫不失联
    monkeypatch.delenv("WATERAGENTS_INTENT_RULES_FILE")
    reload_intent_rules()
