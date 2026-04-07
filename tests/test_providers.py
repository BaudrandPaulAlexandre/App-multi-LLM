"""
test_providers.py
-----------------
Tests unitaires pour les providers LLM.

On utilise des mocks pour ne pas faire de vrais appels API pendant les tests.
Lancez avec : pytest tests/
"""

from unittest.mock import MagicMock, patch

import pytest

from eloquent.providers import (
    GroqProvider,
    LLMResponse,
    QwenOllamaProvider,
    test_determinism,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_mock_completion(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    """Crée un objet completion Groq/OpenAI factice."""
    mock = MagicMock()
    mock.choices[0].message.content = content
    mock.usage.prompt_tokens = prompt_tokens
    mock.usage.completion_tokens = completion_tokens
    return mock


# ---------------------------------------------------------------------------
# Tests GroqProvider
# ---------------------------------------------------------------------------

class TestGroqProvider:

    @patch("eloquent.providers.GroqProvider.__init__", return_value=None)
    def test_generate_returns_llm_response(self, mock_init):
        """generate() doit retourner un LLMResponse bien formé."""
        from groq import Groq
        provider = GroqProvider.__new__(GroqProvider)
        provider._model = "llama-3.1-8b-instant"
        provider._client = MagicMock()
        provider._client.chat.completions.create.return_value = make_mock_completion(
            "Paris est la capitale de la France."
        )

        resp = provider.generate(
            messages=[{"role": "user", "content": "Capitale de la France ?"}],
            temperature=0.0,
            max_tokens=50,
        )

        assert isinstance(resp, LLMResponse)
        assert resp.content == "Paris est la capitale de la France."
        assert resp.provider_name == "groq"
        assert resp.error is None
        assert resp.success is True

    @patch("eloquent.providers.GroqProvider.__init__", return_value=None)
    def test_generate_safe_catches_exception(self, mock_init):
        """generate_safe() ne doit jamais lever d'exception."""
        provider = GroqProvider.__new__(GroqProvider)
        provider._model = "llama-3.1-8b-instant"
        provider._client = MagicMock()
        provider._client.chat.completions.create.side_effect = ConnectionError("timeout")

        resp = provider.generate_safe(
            messages=[{"role": "user", "content": "test"}],
        )

        assert resp.content == ""
        assert resp.error == "timeout"
        assert resp.success is False


# ---------------------------------------------------------------------------
# Tests QwenOllamaProvider
# ---------------------------------------------------------------------------

class TestQwenOllamaProvider:

    @patch("eloquent.providers.QwenOllamaProvider.__init__", return_value=None)
    def test_provider_name(self, mock_init):
        provider = QwenOllamaProvider.__new__(QwenOllamaProvider)
        assert provider.provider_name == "qwen_ollama"

    @patch("eloquent.providers.QwenOllamaProvider.__init__", return_value=None)
    def test_generate_returns_content(self, mock_init):
        provider = QwenOllamaProvider.__new__(QwenOllamaProvider)
        provider._model = "qwen2.5:3b"
        provider._client = MagicMock()
        provider._client.chat.completions.create.return_value = make_mock_completion(
            "La durée légale est de 35 heures."
        )

        resp = provider.generate(
            messages=[{"role": "user", "content": "Durée légale du travail ?"}],
            temperature=0.0,
        )

        assert resp.content == "La durée légale est de 35 heures."
        assert resp.success is True


# ---------------------------------------------------------------------------
# Tests test_determinism()
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_deterministic_provider_returns_true(self):
        """Un provider qui retourne toujours la même réponse est déterministe."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "mock"
        mock_provider.generate_safe.return_value = LLMResponse(
            content="Même réponse",
            model="mock",
            provider_name="mock",
            latency_ms=10.0,
        )

        result = test_determinism(mock_provider, n_runs=2)
        assert result is True

    def test_non_deterministic_provider_returns_false(self):
        """Un provider qui varie ses réponses est non déterministe."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "mock"
        # Deux réponses différentes à chaque appel
        mock_provider.generate_safe.side_effect = [
            LLMResponse("Réponse A", "mock", "mock", 10.0),
            LLMResponse("Réponse B", "mock", "mock", 12.0),
        ]

        result = test_determinism(mock_provider, n_runs=2)
        assert result is False
