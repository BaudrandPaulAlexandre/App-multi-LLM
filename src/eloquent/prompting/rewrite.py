"""
prompting/rewrite.py
--------------------
Lot C — Variante 3 : chaque question est paraphrasée par un LLM tiers
(le "rewriter") avant d'être envoyée au modèle cible.

Le rewriter est un LLMProvider distinct injecté à l'instanciation. En cas
d'échec (timeout, quota, réponse vide), fallback automatique sur le texte
original — la trace le signale via `rewriter_status: "fallback_original"`.
"""

from __future__ import annotations

from eloquent.logger import get_logger
from eloquent.prompting.base import PromptBuildResult

logger = get_logger(__name__)


REWRITE_INSTRUCTION_TEMPLATE = (
    "You are a question normalizer for a multilingual benchmark. "
    "Rewrite the following question in the SAME language. Goals: "
    "(1) remove ambiguity, (2) make the cultural/geographic context explicit "
    "if it was implicit, (3) keep the question concise (one sentence). "
    "Output ONLY the rewritten question, no preamble.\n\n"
    "Original ({lang}): {question}\n"
    "Rewritten:"
)


class RewriteStrategy:
    """
    Configuration YAML :
        prompting:
          strategy: "rewrite"
          rewriter:
            provider: "groq"
            model: "llama-3.1-8b-instant"
            max_tokens: 80
    """
    strategy_name = "rewrite"

    def __init__(self, rewriter, max_tokens: int = 80) -> None:
        # rewriter : LLMProvider déjà instancié — injection explicite pour
        # que le pipeline garde le contrôle du cycle de vie.
        self.rewriter = rewriter
        self.max_tokens = max_tokens

    def build(self, question_text: str, lang: str) -> PromptBuildResult:
        prompt = REWRITE_INSTRUCTION_TEMPLATE.format(
            lang=lang, question=question_text,
        )
        rewriter_resp = self.rewriter.generate_safe(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )

        # Fallback : mieux vaut une question non réécrite qu'une question manquante.
        if not rewriter_resp.success or not rewriter_resp.content.strip():
            logger.warning(
                "Rewriter a échoué (%s) — fallback sur le texte original.",
                rewriter_resp.error or "réponse vide",
            )
            rewritten = question_text
            rewriter_status = "fallback_original"
        else:
            rewritten = rewriter_resp.content.strip()
            rewriter_status = "ok"

        messages = [{"role": "user", "content": rewritten}]
        trace = {
            "strategy": self.strategy_name,
            "lang": lang,
            "rewriter_provider": self.rewriter.provider_name,
            "rewriter_status": rewriter_status,
            "rewriter_latency_ms": round(rewriter_resp.latency_ms, 1),
            "original_text": question_text,
            "rewritten_text": rewritten,
        }
        return PromptBuildResult(
            messages=messages, trace=trace, rewritten_text=rewritten,
        )
