# tests/test_embedding.py

from unittest.mock import MagicMock, patch

import openai
import pytest

from darca_embeddings import (
    BaseEmbeddingClient,
    EmbeddingAPIKeyMissing,
    EmbeddingClient,
    EmbeddingException,
    EmbeddingResponseError,
    OpenAIEmbeddingClient,
)

# -----------------------------------
# 1. Testing the abstract base client
# -----------------------------------


class DummyClient(BaseEmbeddingClient):
    def get_embedding(self, text: str) -> list[float]:
        return [1.0, 2.0, 3.0]


def test_base_embedding_client_get_embeddings():
    dummy = DummyClient()
    inputs = ["hello", "world"]
    outputs = dummy.get_embeddings(inputs)

    assert len(outputs) == 2
    assert outputs[0] == [1.0, 2.0, 3.0]
    assert outputs[1] == [1.0, 2.0, 3.0]


# ------------------------------------------
# 2. Testing OpenAIEmbeddingClient directly
# ------------------------------------------


def test_openai_embedding_client_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EmbeddingAPIKeyMissing):
        _ = OpenAIEmbeddingClient()


@patch("openai.embeddings.create")
def test_openai_embedding_client_success(mock_create, openai_api_key):
    mock_create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
    )
    client = OpenAIEmbeddingClient()  # uses fixture-provided env var
    embedding = client.get_embedding("Hello world")

    assert embedding == [0.1, 0.2, 0.3]
    mock_create.assert_called_once_with(
        input="Hello world", model="text-embedding-ada-002"
    )


@patch("openai.embeddings.create", side_effect=openai.OpenAIError("API Error"))
def test_openai_embedding_client_openai_error(_, openai_api_key):
    client = OpenAIEmbeddingClient()
    with pytest.raises(EmbeddingResponseError) as exc_info:
        client.get_embedding("Hello world")
    assert "OpenAI embedding API returned an error." in str(exc_info.value)


@patch("openai.embeddings.create", side_effect=ValueError("Some other error"))
def test_openai_embedding_client_general_error(_, openai_api_key):
    client = OpenAIEmbeddingClient()
    with pytest.raises(EmbeddingResponseError) as exc_info:
        client.get_embedding("Hello world")
    assert (
        "Unexpected failure during OpenAI embedding response parsing."
        in str(exc_info.value)
    )


def test_base_embedding_client_abstract_pass():
    """
    Force coverage of the abstract method's 'pass' line by
    subclassing BaseEmbeddingClient and calling super().get_embedding().
    """

    class PartialClient(BaseEmbeddingClient):
        # We implement get_embedding in a way that calls the parent's method
        # (which is 'pass'), causing NotImplementedError to be raised.
        def get_embedding(self, text: str) -> list[float]:
            return super().get_embedding(text)

    client = PartialClient()
    with pytest.raises(NotImplementedError):
        client.get_embedding("test")


# -----------------------------------
# 3. Testing the EmbeddingClient wrap
# -----------------------------------


def test_embedding_client_openai_backend(openai_api_key):
    client = EmbeddingClient(backend="openai")
    assert isinstance(client._client, OpenAIEmbeddingClient)


def test_embedding_client_huggingface_backend():
    with pytest.raises(EmbeddingException) as exc_info:
        _ = EmbeddingClient(backend="huggingface")
    assert "is not implemented yet." in str(exc_info.value)


def test_embedding_client_unsupported_backend():
    with pytest.raises(EmbeddingException) as exc_info:
        _ = EmbeddingClient(backend="unsupported")
    assert "is not supported." in str(exc_info.value)


def test_embedding_client_getattr(openai_api_key):
    client = EmbeddingClient(backend="openai")
    with patch.object(
        client._client, "get_embedding", return_value=[0.4, 0.5, 0.6]
    ) as mock_method:
        result = client.get_embedding("Delegation test")
        mock_method.assert_called_once_with("Delegation test")
        assert result == [0.4, 0.5, 0.6]
