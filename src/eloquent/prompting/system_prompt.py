"""
prompting/system_prompt.py
--------------------------
Lot C — Variante 1 : un system prompt unique précède chaque question.

Trois presets prêts à l'emploi (concise / neutral / culturally_aware) ou
prompt entièrement personnalisé via le YAML.
"""

from __future__ import annotations

from eloquent.prompting.base import PromptBuildResult


SYSTEM_PROMPT_LIBRARY: dict[str, str] = {
    "concise": (
        "You are a concise assistant. Always answer in the same language as the "
        "user's question, in a single short sentence (max 25 words). "
        "Do not add disclaimers, greetings, or meta-comments."
    ),
    "neutral": (
        "You are a neutral, factual assistant. Always answer in the same language "
        "as the user's question, in one short sentence. Avoid stereotypes and "
        "personal opinions. State facts only."
    ),
    "culturally_aware": (
        "You are a culturally aware assistant. Always answer in the same language "
        "as the user's question, in one short sentence. When the question implies "
        "a cultural context, ground your answer in that specific context rather "
        "than giving a generic worldwide answer."
    ),
}


class SystemPromptStrategy:
    """
    Configuration YAML :
        prompting:
          strategy: "system_prompt"
          preset: "concise"            # ou "neutral", "culturally_aware"
          # OU surcharge directe :
          # system_prompt: "You are ..."
    """
    strategy_name = "system_prompt"

    def __init__(
        self,
        system_prompt: str | None = None,
        preset: str | None = None,
    ) -> None:
        if system_prompt:
            self.system_prompt = system_prompt
            self.preset = "custom"
        elif preset and preset in SYSTEM_PROMPT_LIBRARY:
            self.system_prompt = SYSTEM_PROMPT_LIBRARY[preset]
            self.preset = preset
        else:
            raise ValueError(
                "SystemPromptStrategy requiert 'system_prompt' ou un 'preset' "
                f"parmi {list(SYSTEM_PROMPT_LIBRARY)}."
            )

    def build(self, question_text: str, lang: str) -> PromptBuildResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question_text},
        ]
        trace = {
            "strategy": self.strategy_name,
            "preset": self.preset,
            "system_prompt": self.system_prompt,
        }
        return PromptBuildResult(messages=messages, trace=trace)
