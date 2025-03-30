import pytest


@pytest.fixture
def openai_api_key(monkeypatch):
    """
    Provides a default environment variable OPENAI_API_KEY for tests
    that need it.
    This fixture is used only when explicitly requested in the test
    function.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    yield
    # Cleanup: remove the var to avoid side effects in parallel tests
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
