"""
prompting.py
------------
Stratégies de construction des messages envoyés au LLM (Lot A + Lot C).

Chaque stratégie prend un texte de question brut (issu du JSONL) et la langue
ISO de la question, puis retourne :
  - une liste de messages prête pour provider.generate()
  - un dict de traçabilité (`trace`) qui décrit exactement la transformation

Stratégies disponibles :
    "vanilla"        : texte brut, pas de system prompt   (Lot A — baseline)
    "system_prompt"  : ajout d'un system prompt unique    (Lot C — variante 1)
    "prefix_suffix"  : préfixe + suffixe par langue       (Lot C — variante 2)
    "rewrite"        : reformulation auto via un LLM tiers (Lot C — variante 3)

Conformité protocole ELOQUENT :
- une question = une session indépendante (pas d'historique)
- la stratégie ne touche jamais aux paramètres de génération (temperature, etc.)
- la traçabilité (`trace`) est sauvegardée à côté de la réponse pour que
  l'analyse Lot D puisse comparer baseline vs variante exactement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from eloquent.logger import get_logger

logger = get_logger(__name__)

Messages = list[dict[str, str]]


# ---------------------------------------------------------------------------
# Résultat enrichi retourné par toutes les stratégies
# ---------------------------------------------------------------------------

@dataclass
class PromptBuildResult:
    """
    Résultat de la construction du prompt.

    - messages       : à passer tel quel à provider.generate()
    - trace          : dict décrivant ce qui a été appliqué (sauvegardé en JSONL)
    - rewritten_text : texte réécrit (variante "rewrite") ou None
    """
    messages: Messages
    trace: dict
    rewritten_text: str | None = None


# ---------------------------------------------------------------------------
# Protocol commun
# ---------------------------------------------------------------------------

class PromptStrategy(Protocol):
    strategy_name: str
    def build(self, question_text: str, lang: str) -> PromptBuildResult: ...


# ---------------------------------------------------------------------------
# Stratégie 1 : Vanilla (Lot A — baseline)
# ---------------------------------------------------------------------------

class VanillaStrategy:
    """Baseline ELOQUENT : texte brut, aucune modification."""
    strategy_name = "vanilla"

    def build(self, question_text: str, lang: str) -> PromptBuildResult:
        messages = [{"role": "user", "content": question_text}]
        trace = {"strategy": self.strategy_name}
        return PromptBuildResult(messages=messages, trace=trace)


# ---------------------------------------------------------------------------
# Stratégie 2 : System prompt (Lot C — variante 1)
# ---------------------------------------------------------------------------

# Banque de system prompts prêts à l'emploi (réponse courte + langue cohérente).
# L'utilisateur peut surcharger via `system_prompt` direct dans le YAML.
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
    Variante 1 — un system prompt unique précédant toutes les questions.

    Configuration YAML :
        prompting:
          strategy: "system_prompt"
          preset: "concise"             # ou "neutral", "culturally_aware"
          # OU surcharge directe :
          system_prompt: "You are ..."
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


# ---------------------------------------------------------------------------
# Stratégie 3 : Prefix + suffix par langue (Lot C — variante 2)
# ---------------------------------------------------------------------------

# Préfixes / suffixes traduits (alignés sur les langues du Lot A).
# Objectif : contraindre le style sans system prompt — utile sur les modèles
# qui suivent moins bien les system prompts (Qwen 3B notamment).
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
    Variante 2 — préfixe + suffixe ajoutés directement au texte de la question.

    Avantages vs system_prompt :
      - fonctionne mieux sur petits modèles ouverts (Qwen 3B, Llama 3B)
      - traduction par langue → renforce la consistance linguistique

    Configuration YAML :
        prompting:
          strategy: "prefix_suffix"
          # facultatif — surcharger les valeurs par défaut :
          prefixes:
            fr: "..."
          suffixes:
            fr: "..."
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


# ---------------------------------------------------------------------------
# Stratégie 4 : Rewrite — reformulation automatique (Lot C — variante 3)
# ---------------------------------------------------------------------------

# Instruction adressée au "rewriter LLM" — pas envoyée au modèle final.
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
    Variante 3 — chaque question est d'abord paraphrasée par un LLM (le
    "rewriter"), puis la version réécrite est envoyée au modèle cible.

    Le rewriter est un LLMProvider distinct (peut être identique au modèle
    cible, ou un modèle plus puissant — ex : Groq llama-70b pour réécrire
    avant d'envoyer à Qwen 3B local).

    Traçabilité : chaque réponse stocke le texte original ET le texte réécrit
    pour que l'analyse Lot D puisse mesurer le delta induit par le rewriter.

    Configuration YAML :
        prompting:
          strategy: "rewrite"
          rewriter:
            provider: "groq"
            model: "llama-3.1-8b-instant"
            temperature: 0.0
            max_tokens: 80
    """
    strategy_name = "rewrite"

    def __init__(self, rewriter, max_tokens: int = 80) -> None:
        # rewriter est un LLMProvider déjà instancié — injection explicite
        # pour que le pipeline garde le contrôle du cycle de vie.
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

        # Si le rewriter échoue, on retombe sur le texte original :
        # mieux vaut une question non réécrite qu'une question manquante.
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_strategy(strategy_name: str, **kwargs) -> PromptStrategy:
    """
    Instancie la bonne stratégie selon le nom passé dans la config YAML.

    kwargs acceptés selon la stratégie :
      vanilla        : (aucun)
      system_prompt  : preset=str | system_prompt=str
      prefix_suffix  : prefixes=dict, suffixes=dict
      rewrite        : rewriter=LLMProvider (obligatoire), max_tokens=int
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
