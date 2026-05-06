"""
prompting/base.py
-----------------
Types et Protocol partagés par toutes les stratégies de prompting.

Une stratégie reçoit (question_text, lang) et renvoie un PromptBuildResult :
  - messages       : prêts pour provider.generate()
  - trace          : dict décrivant la transformation, sauvegardé en JSONL
  - rewritten_text : texte réécrit (variante "rewrite") ou None
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

Messages = list[dict[str, str]]


@dataclass
class PromptBuildResult:
    messages: Messages
    trace: dict
    rewritten_text: str | None = None


class PromptStrategy(Protocol):
    strategy_name: str
    def build(self, question_text: str, lang: str) -> PromptBuildResult: ...
