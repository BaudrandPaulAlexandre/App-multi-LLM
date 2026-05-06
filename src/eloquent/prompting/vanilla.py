"""
prompting/vanilla.py
--------------------
Stratégie baseline (Lot A) : texte brut, aucune transformation.
"""

from __future__ import annotations

from eloquent.prompting.base import PromptBuildResult


class VanillaStrategy:
    """Baseline ELOQUENT : texte brut, aucune modification."""
    strategy_name = "vanilla"

    def build(self, question_text: str, lang: str) -> PromptBuildResult:
        messages = [{"role": "user", "content": question_text}]
        trace = {"strategy": self.strategy_name}
        return PromptBuildResult(messages=messages, trace=trace)
