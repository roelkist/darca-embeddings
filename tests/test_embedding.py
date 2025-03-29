"""
test_embedding.py

Comprehensive tests for darca-embeddings, focusing on
embedding.py to achieve near 100% code coverage.
"""

import pytest
from unittest.mock import MagicMock

import os
import openai

from darca_embeddings.embedding import (
    BaseEmbeddingClient,
    OpenAIEmbeddingClient,
    EmbeddingClient,
    EmbeddingException,
    EmbeddingAPIKeyMissing,
    EmbeddingResponseError
)


# -------------------------------------------------------------------
# Test the abstract base class
# -------------------------------------------------------------------

class MockEmbeddingClient(BaseEmbeddingClient):
    """
    Minimal subclass of BaseEmbeddingClient for testing abstract methods.
    """
    def __init__(self):
        pass

    def get_embedding(self, text: str) -> list[float]:
        return [float(ord(c)) for c in text]  # silly example: convert chars to float

def test_base_embedding_client_get_embeddings():
    """
    Verifies BaseEmbeddingClient.get_embeddings calls get_embedding
    on each text by default.
    """
    client = MockEmbeddingClient()
    texts = ["A", "BC"]
    embeddings = client.get_embeddings(texts)
    assert len(embeddings) == 2
    assert embeddings[0] == [float(ord("A"))]
    assert embeddings[1] == [float(ord("B")), float(ord("C"))]


# -------------------------------------------------------------------
# Test OpenAIEmbeddingClient
# -------------------------------------------------------------------

def test_openai_embedding_client_init_no_api_key(monkeypatch):
    """
    Ensures that if OPENAI_API_KEY is missing, EmbeddingAPIKeyMissing is raised.
    """
    # Remove any existing key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EmbeddingAPIKeyMissing) as exc_info:
        OpenAIEmbeddingClient()
    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_openai_embedding_client_init_success(set_openai_api_key):
    """
    Tests that OpenAIEmbeddingClient initializes with no errors
    when OPENAI_API_KEY is set.
    """
    client = OpenAIEmbeddingClient()
    assert client.model == "text-embedding-ada-002"
    # Also ensure openai.api_key is set in the library
    assert openai.api_key == "test_key"


def test_openai_embedding_client_get_embedding_success(
    set_openai_api_key,
    mock_openai_embedding_create
):
    """
    Tests a successful embedding request using OpenAIEmbeddingClient.
    """
    # Mock a typical OpenAI embedding response
    mock_openai_embedding_create.return_value = {
        "data": [
            {"embedding": [0.123, 0.456, 0.789]}
        ]
    }

    client = OpenAIEmbeddingClient()
    vector = client.get_embedding("Hello world")
    assert vector == [0.123, 0.456, 0.789]
    mock_openai_embedding_create.assert_called_once()


def test_openai_embedding_client_get_embedding_openai_error(
    set_openai_api_key,
    mock_openai_embedding_create
):
    """
    Tests that OpenAIEmbeddingClient raises EmbeddingResponseError
    if openai.Embedding.create raises an OpenAIError.
    """
    from openai.error import OpenAIError

    mock_openai_embedding_create.side_effect = OpenAIError("Some OpenAI issue")

    client = OpenAIEmbeddingClient()
    with pytest.raises(EmbeddingResponseError) as exc_info:
        client.get_embedding("trigger error")
    assert "OpenAI embedding API returned an error." in str(exc_info.value)


def test_openai_embedding_client_get_embedding_unexpected_exception(
    set_openai_api_key,
    mock_openai_embedding_create
):
    """
    Tests that OpenAIEmbeddingClient raises EmbeddingResponseError
    if an unexpected exception occurs.
    """
    mock_openai_embedding_create.side_effect = ValueError("Some random error")

    client = OpenAIEmbeddingClient()
    with pytest.raises(EmbeddingResponseError) as exc_info:
        client.get_embedding("trigger exception")
    assert "Unexpected failure" in str(exc_info.value)


def test_openai_embedding_client_get_embeddings_batch(
    set_openai_api_key,
    mock_openai_embedding_create
):
    """
    Tests that get_embeddings calls get_embedding for each text.
    """
    mock_openai_embedding_create.return_value = {"data": [{"embedding": [1.0]}]}

    client = OpenAIEmbeddingClient()
    results = client.get_embeddings(["A", "B", "C"])
    # We'll get the same result for each call
    assert len(results) == 3
    for res in results:
        assert res == [1.0]
    # openai.Embedding.create was called three times
    assert mock_openai_embedding_create.call_count == 3


# -------------------------------------------------------------------
# Test EmbeddingClient
# -------------------------------------------------------------------

def test_embedding_client_openai_backend(set_openai_api_key):
    """
    Tests that EmbeddingClient initializes an OpenAIEmbeddingClient by default.
    """
    emb_client = EmbeddingClient()
    # Check that it's an OpenAIEmbeddingClient
    assert isinstance(emb_client._client, OpenAIEmbeddingClient)


def test_embedding_client_huggingface_backend():
    """
    Tests that specifying 'huggingface' raises EmbeddingException
    because it's not implemented yet.
    """
    with pytest.raises(EmbeddingException) as exc_info:
        EmbeddingClient(backend="huggingface")
    assert "not implemented yet" in str(exc_info.value)


def test_embedding_client_unknown_backend():
    """
    Tests that specifying an unknown backend raises EmbeddingException.
    """
    with pytest.raises(EmbeddingException) as exc_info:
        EmbeddingClient(backend="unknown")
    assert "is not supported" in str(exc_info.value)


def test_embedding_client_getattr_pass_through(set_openai_api_key):
    """
    Tests that calling a method not on EmbeddingClient is passed
    to the underlying backend client.
    """
    emb_client = EmbeddingClient()
    # We know get_embeddings is on the backend. We'll mock it:
    with pytest.raises(TypeError):
        # get_embeddings expects a list of strings
        emb_client.get_embeddings("Not a list")
