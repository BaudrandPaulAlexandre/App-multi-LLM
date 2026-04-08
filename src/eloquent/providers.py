"""
providers.py
------------
Abstraction multi-LLM : classe de base + deux implémentations.

  - LLMProvider      : interface abstraite commune
  - LLMResponse      : dataclass de résultat standardisé
  - GroqProvider     : appels via API Groq
  - QwenOllamaProvider : Qwen 3B local via Ollama
  - build_provider_from_config() : factory depuis RunConfig
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from eloquent.config import RunConfig
from eloquent.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclass de résultat
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Résultat standardisé retourné par tous les providers."""
    content: str
    model: str
    provider_name: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and len(self.content) > 0


# ---------------------------------------------------------------------------
# Classe de base abstraite
# ---------------------------------------------------------------------------

class LLMProvider(ABC):

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 150,
        **kwargs: Any,
    ) -> LLMResponse: ...

    @abstractmethod
    def health_check(self) -> bool: ...

    def generate_safe(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 150,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Wrapper avec gestion d'erreur : ne lève jamais d'exception.
        Le pipeline peut continuer même si une question échoue.
        L'erreur est loguée et stockée dans LLMResponse.error.
        """
        try:
            return self.generate(messages, temperature, max_tokens, **kwargs)
        except Exception as exc:
            logger.error("[%s] Erreur generate : %s", self.provider_name, exc)
            return LLMResponse(
                content="",
                model="unknown",
                provider_name=self.provider_name,
                latency_ms=0.0,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Provider 1 : Groq
# ---------------------------------------------------------------------------

class GroqProvider(LLMProvider):
    """
    Appelle l'API Groq.

    Modèles disponibles :
        llama-3.1-8b-instant    (rapide, recommandé pour la baseline)
        llama-3.3-70b-versatile (plus puissant, plus lent)
        mixtral-8x7b-32768

    Docs : https://console.groq.com/docs/models
    """

    def __init__(self, model: str, api_key: str) -> None:
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError("pip install groq") from e

        self._model = model
        self._client = Groq(api_key=api_key)
        logger.info("GroqProvider prêt — modèle : %s", model)

    @property
    def provider_name(self) -> str:
        return "groq"

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 150,
        **kwargs: Any,
    ) -> LLMResponse:
        t0 = time.perf_counter()

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        content = completion.choices[0].message.content or ""
        usage = completion.usage

        logger.debug(
            "[groq] %.0fms | in=%s | out=%s tok",
            latency_ms,
            usage.prompt_tokens if usage else "?",
            usage.completion_tokens if usage else "?",
        )

        return LLMResponse(
            content=content,
            model=self._model,
            provider_name=self.provider_name,
            latency_ms=latency_ms,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )

    def health_check(self) -> bool:
        try:
            resp = self.generate(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            if resp.success:
                logger.info("[groq] health_check ✓")
                return True
            logger.warning("[groq] health_check : réponse vide")
            return False
        except Exception as exc:
            logger.error("[groq] health_check KO : %s", exc)
            return False


# ---------------------------------------------------------------------------
# Provider 2 : Qwen via Ollama
# ---------------------------------------------------------------------------

class QwenOllamaProvider(LLMProvider):
    """
    Appelle Qwen 2.5 3B via Ollama (API REST locale compatible OpenAI).

    Prérequis :
        1. Installer Ollama : https://ollama.com/download
        2. Télécharger le modèle : ollama pull qwen2.5:3b
        3. Le serveur démarre automatiquement (ou : ollama serve)

    Note : fonctionne 100% hors-ligne une fois le modèle téléchargé.
    Note déterminisme : Ollama/llama.cpp est strictement déterministe à
    temperature=0 — les deux runs produiront exactement la même réponse.
    """

    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e

        self._model = model
        self._base_url = base_url
        # "ollama" est un placeholder : Ollama n'exige pas de vraie clé API
        self._client = OpenAI(base_url=base_url, api_key="ollama")
        logger.info(
            "QwenOllamaProvider prêt — modèle : %s sur %s", model, base_url
        )

    @property
    def provider_name(self) -> str:
        return "qwen_ollama"

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 150,
        **kwargs: Any,
    ) -> LLMResponse:
        t0 = time.perf_counter()

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        content = completion.choices[0].message.content or ""
        usage = completion.usage

        logger.debug(
            "[qwen_ollama] %.0fms | in=%s | out=%s tok",
            latency_ms,
            usage.prompt_tokens if usage else "?",
            usage.completion_tokens if usage else "?",
        )

        return LLMResponse(
            content=content,
            model=self._model,
            provider_name=self.provider_name,
            latency_ms=latency_ms,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )

    def health_check(self) -> bool:
        try:
            resp = self.generate(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            if resp.success:
                logger.info("[qwen_ollama] health_check ✓")
                return True
            logger.warning("[qwen_ollama] health_check : réponse vide")
            return False
        except Exception as exc:
            logger.error(
                "[qwen_ollama] health_check KO : %s\n"
                "  → Vérifiez qu'Ollama tourne : ollama serve\n"
                "  → Modèle disponible : ollama pull %s",
                exc, self._model,
            )
            return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_provider_from_config(cfg: RunConfig) -> LLMProvider:
    """
    Instancie le bon provider depuis un RunConfig chargé.
    C'est le seul endroit où le type de provider est résolu.
    """
    if cfg.provider == "groq":
        return GroqProvider(model=cfg.model, api_key=cfg.groq_api_key)

    if cfg.provider == "qwen_ollama":
        return QwenOllamaProvider(
            model=cfg.model,
            base_url=cfg.ollama_base_url,
        )

    raise ValueError(
        f"Provider inconnu : '{cfg.provider}'. "
        f"Valeurs acceptées : 'groq', 'qwen_ollama'."
    )


# ---------------------------------------------------------------------------
# Test de déterminisme
# ---------------------------------------------------------------------------

def test_determinism(
    provider: LLMProvider,
    question: str = "What is the capital of France?",
    n_runs: int = 2,
) -> bool:
    """
    Envoie la même question n_runs fois à temperature=0.
    Vérifie que toutes les réponses sont identiques.

    Retourne True si déterministe, False sinon.
    Utilisé dans run.py avant le lancement de la baseline.
    """
    messages = [{"role": "user", "content": question}]
    responses: list[str] = []

    logger.info(
        "Test déterminisme — %d appels sur [%s]",
        n_runs, provider.provider_name,
    )

    for i in range(n_runs):
        resp = provider.generate_safe(messages, temperature=0.0, max_tokens=100)
        responses.append(resp.content)
        logger.info("  Run %d/%d → «%s»", i + 1, n_runs, resp.content[:100])

    is_deterministic = len(set(responses)) == 1

    if is_deterministic:
        logger.info("✅ [%s] DÉTERMINISTE", provider.provider_name)
    else:
        logger.warning(
            "⚠️  [%s] NON DÉTERMINISTE — %d réponses distinctes",
            provider.provider_name, len(set(responses)),
        )

    return is_deterministic
