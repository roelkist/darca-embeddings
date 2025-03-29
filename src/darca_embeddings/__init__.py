"""
darca-embeddings

Modular, backend-agnostic interface for generating text embeddings.
Default backend: OpenAI

Exports:
    - EmbeddingClient: Unified interface for embeddings
    - BaseEmbeddingClient: Abstract base for all embeddings backends
    - OpenAIEmbeddingClient: Concrete backend using OpenAI Embedding API
    - All custom exceptions
"""

from .embedding import (
    EmbeddingClient,
    BaseEmbeddingClient,
    OpenAIEmbeddingClient,
    EmbeddingException,
    EmbeddingAPIKeyMissing,
    EmbeddingResponseError
)

__all__ = [
    "EmbeddingClient",
    "BaseEmbeddingClient",
    "OpenAIEmbeddingClient",
    "EmbeddingException",
    "EmbeddingAPIKeyMissing",
    "EmbeddingResponseError"
]
