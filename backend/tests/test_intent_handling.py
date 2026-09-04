"""概念解释 / 实时研判意图区分回归测试。

验证通用机制：
1. 直接回答（direct_chat）提示词能处理领域概念解释，而不是只当闲聊。
2. 工具规划提示词把"概念解释/知识问答"与"实时数据/任务"区分开，并要求概念类不调用工具。
3. 预警级别解读技能的配置也明确区分两类问题，避免把"解释四级含义"误解为"研判当前等级"。
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.graph.nodes import _plan_via_function_calling
from agent.prompts import DIRECT_CHAT_PROMPT


def _mock_planner_llm_no_tools():
    """构造 planner 返回空 tool_calls 的 mock，捕获 create 调用参数。"""
    create_mock = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.tool_calls = None
    # 救援解析会读取正文（FC 为空时尝试从文本抢救工具调用）
    resp.choices[0].message.content = ""
    create_mock.chat.completions.create.return_value = resp

    client_mock = MagicMock()
    client_mock.with_options.return_value = create_mock
    return client_mock, create_mock


class TestDirectChatPrompt:
    def test_prompt_handles_concept_questions(self):
        """direct_chat 提示词应包含概念解释分支，而不是只面向闲聊。"""
        assert "概念解释" in DIRECT_CHAT_PROMPT
        assert "四级预警分别是什么含义" in DIRECT_CHAT_PROMPT
        assert "不要主动调用工具" in DIRECT_CHAT_PROMPT


class TestPlannerPrompt:
    def test_prompt_has_general_concept_rule(self):
        """planner 提示词应把概念解释类判定为无需工具，且规则对任意问题通用。"""
        client_mock, create_mock = _mock_planner_llm_no_tools()
        with patch("agent.graph.nodes.get_llm_client", return_value=client_mock), \
             patch("agent.graph.nodes.get_llm_config",
                   return_value={"model": "test", "max_tool_rounds": 5}):
            _plan_via_function_calling("预警等级怎么划分", "")

        system = create_mock.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "概念解释" in system
        assert "返回空工具调用列表" in system
        assert "即使当前激活的 Skill 指令里写有工具流程" in system


class TestWarningLevelSkillConfig:
    def test_skill_distinguishes_concept_and_realtime(self):
        """预警级别解读技能应明确区分概念解释与实时研判两类问题。"""
        path = Path(__file__).resolve().parent.parent / "data" / "skills.json"
        skills = json.loads(path.read_text(encoding="utf-8"))["skills"]
        skill = next(s for s in skills if s["name"] == "warning_level_interpretation")

        assert "概念解释类" in skill["instructions"]
        assert "不要调用任何工具" in skill["instructions"]
        assert "实时研判类" in skill["instructions"]
