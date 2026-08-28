"""知识问答种子和合成器测试。"""
from train.data_gen.knowledge_synthesizer import _KNOWLEDGE_SYSTEM_PROMPT, synthesize_knowledge_one
from train.data_gen.seed_queries import SeedQuery, get_knowledge_seeds, get_seeds


def test_get_seeds_excludes_knowledge():
    """get_seeds 只返回业务种子，不含知识问答。"""
    seeds = get_seeds()
    assert len(seeds) == 30
    assert all(s.intent != "knowledge" for s in seeds)


def test_get_knowledge_seeds():
    """get_knowledge_seeds 返回知识问答种子。"""
    seeds = get_knowledge_seeds()
    assert len(seeds) == 15
    assert all(s.intent == "knowledge" for s in seeds)
    # 知识问答 level 为空
    assert all(s.level == "" for s in seeds)


def test_knowledge_seed_fields():
    """知识问答种子字段完整。"""
    seeds = get_knowledge_seeds()
    for s in seeds:
        assert s.query
        assert s.intent == "knowledge"
        assert s.level == ""
        # station 可为空（纯概念问答）或非空（站点常识）


def test_knowledge_synthesizer_prompt():
    """知识合成器有独立的 system prompt。"""
    assert "防汛" in _KNOWLEDGE_SYSTEM_PROMPT
    assert "知识性问题" in _KNOWLEDGE_SYSTEM_PROMPT


def test_synthesize_knowledge_one_with_mock_client():
    """用 mock 客户端测试知识问答合成。"""
    class MockMessage:
        def __init__(self, content):
            self.content = content

    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)

    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    class MockClient:
        def __init__(self, answer: str):
            self.chat = self
            self.completions = self
            self._answer = answer

        def create(self, **kwargs):
            return MockResponse(self._answer)

    seed = SeedQuery("什么是防汛？", "", "", "knowledge")
    client = MockClient("防汛是指为防止洪水灾害而采取的各项措施。")
    result = synthesize_knowledge_one(client, "test-model", seed)
    assert result is not None
    assert len(result["messages"]) == 3
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][1]["role"] == "user"
    assert result["messages"][2]["role"] == "assistant"
    assert "防汛" in result["messages"][2]["content"]


def test_synthesize_knowledge_one_empty_answer():
    """空回答返回 None。"""
    class MockMessage:
        def __init__(self, content):
            self.content = content

    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)

    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    class MockClient:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            return MockResponse("")

    seed = SeedQuery("什么是防汛？", "", "", "knowledge")
    result = synthesize_knowledge_one(MockClient(), "test-model", seed)
    assert result is None
