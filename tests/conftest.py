"""
conftest.py for darca-embeddings

This file holds global pytest configurations, fixtures,
and other shared utilities for testing the darca-embeddings
package.

Docs:
    https://docs.pytest.org/en/stable/fixture.html
"""

import os
import pytest
from unittest.mock import patch


@pytest.fixture
def set_openai_api_key(monkeypatch):
    """
    Fixture to set a dummy OPENAI_API_KEY environment variable
    before each test. Automatically removes it afterward.

    Usage:
        def test_something(set_openai_api_key):
            # OPENAI_API_KEY is set
            ...
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    yield
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def mock_openai_embedding_create():
    """
    Fixture to patch openai.Embedding.create and yield a mock object.
    Allows tests to simulate success/failure responses from OpenAI.
    """
    with patch("openai.Embedding.create") as mock_create:
        yield mock_create
