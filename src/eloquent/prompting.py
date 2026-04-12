"""
prompting.py
------------
Stratégies de construction des messages envoyés au LLM.

Chaque stratégie prend un texte de question brut (issu du JSONL)
et retourne une liste de messages prête à être passée à provider.generate().

Stratégies disponibles :
    "vanilla"       : texte brut, pas de system prompt (baseline obligatoire)
    "system_prompt" : ajout d'un system prompt de style (Lot C)
    "rewrite"       : reformulation automatique via un second LLM (Lot C, avancé)

Le Lot A n'implémente que "vanilla". Les autres sont des stubs prêts
à être complétés par le Lot C.
"""

from __future__ import annotations

from typing import Protocol

from eloquent.logger import get_logger

logger = get_logger(__name__)

# Type alias pour une liste de messages chat
Messages = list[dict[str, str]]


# ---------------------------------------------------------------------------
# Protocol (interface duck-typing) pour une stratégie de prompting
# ---------------------------------------------------------------------------

class PromptStrategy(Protocol):
    """
    Une stratégie de prompting transforme un texte de question
    en liste de messages chat.
    """
    def build_messages(self, question_text: str) -> Messages: ...
    strategy_name: str


# ---------------------------------------------------------------------------
# Stratégie 1 : Vanilla (baseline obligatoire)
# ---------------------------------------------------------------------------

class VanillaStrategy:
    """
    Stratégie baseline : envoie le texte brut de la question sans aucun
    system prompt ni transformation.

    Conforme au protocole ELOQUENT :
    - une question = une session indépendante (pas d'historique)
    - pas de prompt engineering
    - réponse courte contrôlée via max_tokens dans la config
    """
    strategy_name = "vanilla"

    def build_messages(self, question_text: str) -> Messages:
        # Session indépendante = une seule paire user/assistant, pas de system
        return [
            {"role": "user", "content": question_text}
        ]


# ---------------------------------------------------------------------------
# Stratégie 2 : System prompt (stub Lot C)
# ---------------------------------------------------------------------------

class SystemPromptStrategy:
    """
    Ajoute un system prompt avant la question.
    Exemple d'usage Lot C : contraindre la langue de réponse,
    imposer la concision, neutraliser le biais culturel, etc.

    À compléter dans le Lot C.
    """
    strategy_name = "system_prompt"

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def build_messages(self, question_text: str) -> Messages:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question_text},
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_strategy(strategy_name: str, **kwargs) -> PromptStrategy:
    """
    Retourne la bonne stratégie selon le nom défini dans la config YAML.

    Args:
        strategy_name : "vanilla" | "system_prompt"
        **kwargs      : paramètres spécifiques à la stratégie
                        (ex: system_prompt="..." pour system_prompt)
    """
    if strategy_name == "vanilla":
        return VanillaStrategy()

    if strategy_name == "system_prompt":
        system_prompt = kwargs.get("system_prompt", "")
        if not system_prompt:
            logger.warning(
                "Stratégie 'system_prompt' sans system_prompt défini — "
                "comportement identique à vanilla."
            )
        return SystemPromptStrategy(system_prompt=system_prompt)

    raise ValueError(
        f"Stratégie inconnue : '{strategy_name}'. "
        f"Valeurs acceptées : 'vanilla', 'system_prompt'."
    )
