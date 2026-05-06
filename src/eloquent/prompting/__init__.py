"""
prompting/
----------
Stratégies de construction des messages envoyés au LLM (Lot A + Lot C).

Chaque stratégie vit dans son propre module pour faciliter les modifications :
    base.py            — types partagés (PromptBuildResult, PromptStrategy)
    vanilla.py         — VanillaStrategy           (Lot A — baseline)
    system_prompt.py   — SystemPromptStrategy      (Lot C — variante 1)
    prefix_suffix.py   — PrefixSuffixStrategy      (Lot C — variante 2)
    rewrite.py         — RewriteStrategy           (Lot C — variante 3)

La factory générique `build_strategy()` instancie la bonne stratégie depuis
le nom donné dans la config YAML — c'est le seul point d'entrée utilisé par
le pipeline.

Conformité protocole ELOQUENT :
- une question = une session indépendante (pas d'historique)
- la stratégie ne touche jamais aux paramètres de génération
- la traçabilité (`trace`) est sauvegardée à côté de la réponse
"""

from __future__ import annotations

from eloquent.prompting.base import (
    Messages,
    PromptBuildResult,
    PromptStrategy,
)
from eloquent.prompting.prefix_suffix import (
    DEFAULT_PREFIXES,
    DEFAULT_SUFFIXES,
    PrefixSuffixStrategy,
)
from eloquent.prompting.rewrite import (
    REWRITE_INSTRUCTION_TEMPLATE,
    RewriteStrategy,
)
from eloquent.prompting.system_prompt import (
    SYSTEM_PROMPT_LIBRARY,
    SystemPromptStrategy,
)
from eloquent.prompting.vanilla import VanillaStrategy

__all__ = [
    "Messages",
    "PromptBuildResult",
    "PromptStrategy",
    "VanillaStrategy",
    "SystemPromptStrategy",
    "SYSTEM_PROMPT_LIBRARY",
    "PrefixSuffixStrategy",
    "DEFAULT_PREFIXES",
    "DEFAULT_SUFFIXES",
    "RewriteStrategy",
    "REWRITE_INSTRUCTION_TEMPLATE",
    "build_strategy",
]


def build_strategy(strategy_name: str, **kwargs) -> PromptStrategy:
    """
    Factory générique : instancie la bonne stratégie depuis son nom.

    kwargs acceptés selon la stratégie :
      vanilla        : (aucun)
      system_prompt  : preset=str | system_prompt=str
      prefix_suffix  : prefixes=dict, suffixes=dict
      rewrite        : rewriter=LLMProvider (obligatoire), max_tokens=int

    Pour ajouter une nouvelle variante : créer un module ``prompting/<nom>.py``
    avec la classe ``XxxStrategy``, l'importer plus haut et ajouter une branche
    ``elif strategy_name == "xxx": ...`` ci-dessous.
    """
    if strategy_name == "vanilla":
        return VanillaStrategy()

    if strategy_name == "system_prompt":
        return SystemPromptStrategy(
            system_prompt=kwargs.get("system_prompt"),
            preset=kwargs.get("preset"),
        )

    if strategy_name == "prefix_suffix":
        return PrefixSuffixStrategy(
            prefixes=kwargs.get("prefixes"),
            suffixes=kwargs.get("suffixes"),
        )

    if strategy_name == "rewrite":
        rewriter = kwargs.get("rewriter")
        if rewriter is None:
            raise ValueError(
                "Stratégie 'rewrite' requiert un argument 'rewriter' "
                "(LLMProvider déjà instancié)."
            )
        return RewriteStrategy(
            rewriter=rewriter,
            max_tokens=kwargs.get("max_tokens", 80),
        )

    raise ValueError(
        f"Stratégie inconnue : '{strategy_name}'. "
        f"Valeurs acceptées : 'vanilla', 'system_prompt', "
        f"'prefix_suffix', 'rewrite'."
    )
