"""
prompting/prefix_suffix.py
--------------------------
Lot C — Variante 2 : préfixe + suffixe par langue, injectés directement
dans le texte utilisateur.

Cible les petits modèles ouverts (Qwen 3B, Llama 3B) qui suivent moins bien
les system prompts qu'un wrapping textuel dans la même langue que la question.
"""

from __future__ import annotations

from eloquent.prompting.base import PromptBuildResult


DEFAULT_PREFIXES: dict[str, str] = {
    "fr": "Réponds en une seule phrase courte, en français : ",
    "en": "Answer in one short sentence, in English: ",
    "es": "Responde en una sola frase corta, en español: ",
    "it": "Rispondi in una sola frase breve, in italiano: ",
    "de": "Antworte in einem einzigen kurzen Satz auf Deutsch: ",
}

DEFAULT_SUFFIXES: dict[str, str] = {
    "fr": "\n\nRéponse (une phrase) :",
    "en": "\n\nAnswer (one sentence):",
    "es": "\n\nRespuesta (una frase):",
    "it": "\n\nRisposta (una frase):",
    "de": "\n\nAntwort (ein Satz):",
}


class PrefixSuffixStrategy:
    """
    Configuration YAML :
        prompting:
          strategy: "prefix_suffix"
          # facultatif — surcharger les valeurs par défaut :
          # prefixes:
          #   fr: "..."
          # suffixes:
          #   fr: "..."
    """
    strategy_name = "prefix_suffix"

    def __init__(
        self,
        prefixes: dict[str, str] | None = None,
        suffixes: dict[str, str] | None = None,
    ) -> None:
        # Merge utilisateur > défauts (l'utilisateur peut n'override qu'une langue)
        self.prefixes = {**DEFAULT_PREFIXES, **(prefixes or {})}
        self.suffixes = {**DEFAULT_SUFFIXES, **(suffixes or {})}

    def build(self, question_text: str, lang: str) -> PromptBuildResult:
        prefix = self.prefixes.get(lang, "")
        suffix = self.suffixes.get(lang, "")
        wrapped = f"{prefix}{question_text}{suffix}"

        messages = [{"role": "user", "content": wrapped}]
        trace = {
            "strategy": self.strategy_name,
            "lang": lang,
            "prefix": prefix,
            "suffix": suffix,
        }
        return PromptBuildResult(messages=messages, trace=trace)
