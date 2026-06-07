"""
provider.py
-----------
Provider OpenRouter -> Groq, isolé du package eloquent.

OpenRouter expose une API OpenAI-compatible. On l'utilise donc exactement
comme le QwenOllamaProvider (client `openai`), à deux différences près :

  1. base_url = https://openrouter.ai/api/v1
  2. on injecte `extra_body={"provider": {...}}` pour FORCER l'inférence
     chez Groq via le provider routing d'OpenRouter.

Ce fichier ne modifie aucun fichier du package eloquent. Il définit une
implémentation autonome de l'interface LLMProvider.

>>> DOSSIER JETABLE : supprimer experiments/openrouter_groq/ retire tout. <<<
"""

from __future__ import annotations

import time
from typing import Any

from eloquent.providers import LLMProvider, LLMResponse
from eloquent.logger import get_logger

logger = get_logger(__name__)


class OpenRouterGroqProvider(LLMProvider):
    """
    Appelle OpenRouter en forçant le fournisseur d'inférence sous-jacent.

    Args:
        model            : id du modèle côté OpenRouter
                           (ex: "meta-llama/llama-3.1-8b-instruct")
        api_key          : clé OpenRouter (OPENROUTER_API_KEY)
        base_url         : endpoint OpenAI-compatible d'OpenRouter
        provider_order   : ordre des fournisseurs imposés (ex: ["Groq"])
        allow_fallbacks  : si False, échoue plutôt que de basculer sur un
                           autre fournisseur que ceux de provider_order
        request_delay_s  : délai appliqué AVANT chaque appel (lisse le débit
                           pour limiter les 429). Non compté dans la latence.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        provider_order: list[str] | None = None,
        allow_fallbacks: bool = False,
        request_delay_s: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY introuvable. "
                "Ajoutez-la dans votre fichier .env (à la racine du projet)."
            )

        self._model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._request_delay_s = max(0.0, request_delay_s)

        # Body supplémentaire injecté à chaque appel : le routing OpenRouter.
        self._provider_routing = {
            "order": provider_order or ["Groq"],
            "allow_fallbacks": allow_fallbacks,
        }
        logger.info(
            "OpenRouterGroqProvider prêt — modèle : %s | routing : %s | délai : %.2fs",
            model, self._provider_routing, self._request_delay_s,
        )

    @property
    def provider_name(self) -> str:
        return "openrouter_groq"

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 150,
        **kwargs: Any,
    ) -> LLMResponse:
        # Throttle volontaire AVANT le chrono : lisse le débit pour limiter
        # les 429, sans gonfler la latence mesurée.
        if self._request_delay_s:
            time.sleep(self._request_delay_s)

        t0 = time.perf_counter()

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"provider": self._provider_routing},
            **kwargs,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        content = completion.choices[0].message.content or ""
        usage = completion.usage

        # OpenRouter renvoie le fournisseur réellement utilisé dans la réponse.
        used = getattr(completion, "provider", None)
        logger.debug(
            "[openrouter_groq] %.0fms | provider=%s | in=%s | out=%s tok",
            latency_ms,
            used or "?",
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
                logger.info("[openrouter_groq] health_check ✓")
                return True
            logger.warning("[openrouter_groq] health_check : réponse vide")
            return False
        except Exception as exc:
            logger.error(
                "[openrouter_groq] health_check KO : %s\n"
                "  → Vérifiez OPENROUTER_API_KEY dans .env\n"
                "  → Vérifiez que le modèle '%s' est routable chez Groq",
                exc, self._model,
            )
            return False
