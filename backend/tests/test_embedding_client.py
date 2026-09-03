"""Embedding 独立凭证测试。

验证 EMBEDDING_API_KEY / EMBEDDING_BASE_URL 与推理 LLM_API_KEY / LLM_BASE_URL
的隔离与回退：配置独立凭证时 embedding 客户端用自己的；留空时回退推理配置。
"""
import pytest

from app.core.config import get_settings
from app.core.llm import get_embedding_client, get_llm_client


@pytest.fixture(autouse=True)
def _reset_singletons():
    """每个用例前后清空单例缓存，避免测试环境变量泄漏到其他用例。"""
    get_settings.cache_clear()
    get_llm_client.cache_clear()
    get_embedding_client.cache_clear()
    yield
    get_settings.cache_clear()
    get_llm_client.cache_clear()
    get_embedding_client.cache_clear()


class TestEmbeddingClientCredentials:
    def test_uses_own_credentials_when_configured(self, monkeypatch):
        """配置独立凭证时，embedding 客户端使用自己的 key/base_url，
        推理客户端不受影响。"""
        monkeypatch.setenv("LLM_API_KEY", "sk-llm-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com")
        monkeypatch.setenv("EMBEDDING_API_KEY", "sk-embed-key")
        monkeypatch.setenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

        embed_client = get_embedding_client()
        assert embed_client.api_key == "sk-embed-key"
        assert "dashscope.aliyuncs.com" in str(embed_client.base_url)

        llm_client = get_llm_client()
        assert llm_client.api_key == "sk-llm-key"
        assert "api.deepseek.com" in str(llm_client.base_url)

    def test_falls_back_to_llm_when_empty(self, monkeypatch):
        """独立凭证留空时回退到 LLM 配置（向后兼容）。"""
        monkeypatch.setenv("LLM_API_KEY", "sk-llm-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com")
        monkeypatch.setenv("EMBEDDING_API_KEY", "")
        monkeypatch.setenv("EMBEDDING_BASE_URL", "")

        embed_client = get_embedding_client()
        assert embed_client.api_key == "sk-llm-key"
        assert "api.deepseek.com" in str(embed_client.base_url)

    def test_embedding_batch_uses_embedding_client(self, monkeypatch):
        """_embed_batch 经 get_embedding_client 发请求（不再复用推理客户端）。"""
        from agent.rag import embedding

        monkeypatch.setenv("LLM_API_KEY", "sk-llm-key")
        monkeypatch.setenv("EMBEDDING_API_KEY", "sk-embed-key")
        monkeypatch.setenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

        captured = {}

        class FakeEmbeddings:
            def create(self, model, input):
                captured["model"] = model
                captured["input"] = input

                class Item:
                    def __init__(self, index):
                        self.index = index
                        self.embedding = [0.1, 0.2]

                class Resp:
                    data = [Item(0), Item(1)]

                return Resp()

        class FakeClient:
            embeddings = FakeEmbeddings()

        monkeypatch.setattr(embedding, "get_embedding_client", lambda: FakeClient())
        result = embedding._embed_batch(["文本A", "文本B"])
        assert captured["model"] == get_settings().LLM_EMBEDDING_MODEL
        assert captured["input"] == ["文本A", "文本B"]
        assert result == [[0.1, 0.2], [0.1, 0.2]]
