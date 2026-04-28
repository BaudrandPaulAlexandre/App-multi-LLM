"""
config.py
---------
Chargement et validation de la configuration YAML d'un run.

Usage :
    from eloquent.config import RunConfig, load_config
    cfg = load_config("configs/baseline_groq.yaml")
    print(cfg.model)          # "llama-3.1-8b-instant"
    print(cfg.generation)     # GenerationParams(temperature=0.0, ...)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

from eloquent.logger import get_logger

logger = get_logger(__name__)

# Charge automatiquement le fichier .env à l'import du module
load_dotenv()


# ---------------------------------------------------------------------------
# Dataclasses de configuration (une par section du YAML)
# ---------------------------------------------------------------------------

@dataclass
class GenerationParams:
    temperature: float = 0.0
    max_tokens: int = 150
    top_p: float = 1.0

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }


@dataclass
class RewriterConfig:
    """Sous-config pour la variante 'rewrite' : LLM utilisé pour réécrire."""
    provider: str = "groq"
    model: str = "llama-3.1-8b-instant"
    max_tokens: int = 80
    ollama_base_url: str = "http://localhost:11434/v1"


@dataclass
class PromptingParams:
    """
    Paramètres de la stratégie de prompting.

    Stratégies supportées (cf. prompting.py) :
      "vanilla"        : Lot A — baseline
      "system_prompt"  : Lot C — variante 1
      "prefix_suffix"  : Lot C — variante 2
      "rewrite"        : Lot C — variante 3
    """
    strategy: str = "vanilla"

    # Pour "system_prompt" — l'un OU l'autre suffit
    preset: str | None = None
    system_prompt: str | None = None

    # Pour "prefix_suffix" — None = utilise les défauts par langue
    prefixes: dict[str, str] | None = None
    suffixes: dict[str, str] | None = None

    # Pour "rewrite"
    rewriter: RewriterConfig | None = None


@dataclass
class PathsConfig:
    input_dir: Path = Path("data/input")
    output_dir: Path = Path("data/output/runs")


@dataclass
class RunConfig:
    """
    Configuration complète d'un run ELOQUENT.
    Correspond exactement à la structure des fichiers YAML dans configs/.
    """
    run_id: str
    provider: str                         # "groq" | "qwen_ollama"
    model: str
    languages: list[str]
    dataset_type: str                     # "specific" | "unspecific"
    generation: GenerationParams
    prompting: PromptingParams
    paths: PathsConfig

    # Champs optionnels selon le provider
    groq_api_key: str | None = None       # Lu depuis .env si provider=groq
    ollama_base_url: str = "http://localhost:11434/v1"

    # Échantillonnage reproductible (aligné sur les runs Lot A déjà produits) :
    # - max_questions = N : tire N questions au hasard par langue
    # - sample_seed         : graine pour rendre le tirage reproductible
    # Si max_questions est None : on prend tout le fichier (pas d'échantillonnage).
    max_questions: int | None = None
    sample_seed: int = 42

    def validate(self) -> None:
        """
        Vérifie la cohérence de la config.
        Lève ValueError avec un message explicite si quelque chose cloche.
        """
        valid_providers = {"groq", "qwen_ollama"}
        if self.provider not in valid_providers:
            raise ValueError(
                f"provider='{self.provider}' invalide. "
                f"Valeurs acceptées : {valid_providers}"
            )

        valid_dataset_types = {"specific", "unspecific"}
        if self.dataset_type not in valid_dataset_types:
            raise ValueError(
                f"dataset_type='{self.dataset_type}' invalide. "
                f"Valeurs acceptées : {valid_dataset_types}"
            )

        if not self.languages:
            raise ValueError("La liste 'languages' ne peut pas être vide.")

        if self.provider == "groq" and not self.groq_api_key:
            raise ValueError(
                "provider=groq mais GROQ_API_KEY introuvable. "
                "Vérifiez votre fichier .env."
            )

        if self.generation.temperature < 0 or self.generation.temperature > 2:
            raise ValueError(
                f"temperature={self.generation.temperature} hors bornes [0, 2]."
            )

        if self.max_questions is not None and self.max_questions <= 0:
            raise ValueError(
                f"max_questions={self.max_questions} doit être un entier "
                f"positif (ou null pour utiliser tout le fichier)."
            )

        valid_strategies = {"vanilla", "system_prompt", "prefix_suffix", "rewrite"}
        if self.prompting.strategy not in valid_strategies:
            raise ValueError(
                f"prompting.strategy='{self.prompting.strategy}' invalide. "
                f"Valeurs acceptées : {valid_strategies}"
            )

        if self.prompting.strategy == "system_prompt":
            if not (self.prompting.preset or self.prompting.system_prompt):
                raise ValueError(
                    "strategy='system_prompt' requiert 'preset' ou 'system_prompt'."
                )

        if self.prompting.strategy == "rewrite" and self.prompting.rewriter is None:
            raise ValueError(
                "strategy='rewrite' requiert une section 'rewriter' dans 'prompting'."
            )

        logger.info("Config '%s' validée ✓", self.run_id)

    def to_dict(self) -> dict:
        """Sérialise la config en dict (pour la sauvegarder dans le run)."""
        prompting_dict: dict = {"strategy": self.prompting.strategy}
        if self.prompting.preset is not None:
            prompting_dict["preset"] = self.prompting.preset
        if self.prompting.system_prompt is not None:
            prompting_dict["system_prompt"] = self.prompting.system_prompt
        if self.prompting.prefixes is not None:
            prompting_dict["prefixes"] = self.prompting.prefixes
        if self.prompting.suffixes is not None:
            prompting_dict["suffixes"] = self.prompting.suffixes
        if self.prompting.rewriter is not None:
            prompting_dict["rewriter"] = {
                "provider": self.prompting.rewriter.provider,
                "model": self.prompting.rewriter.model,
                "max_tokens": self.prompting.rewriter.max_tokens,
            }

        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "languages": self.languages,
            "dataset_type": self.dataset_type,
            "generation": self.generation.to_dict(),
            "prompting": prompting_dict,
            "ollama_base_url": self.ollama_base_url,
            "max_questions": self.max_questions,
            "sample_seed": self.sample_seed,
            # On ne sérialise JAMAIS la clé API
        }


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _load_prompting(raw: dict) -> PromptingParams:
    """Construit un PromptingParams depuis la section 'prompting' du YAML."""
    rewriter_raw = raw.get("rewriter")
    rewriter_cfg = None
    if rewriter_raw:
        rewriter_cfg = RewriterConfig(
            provider=rewriter_raw.get("provider", "groq"),
            model=rewriter_raw.get("model", "llama-3.1-8b-instant"),
            max_tokens=rewriter_raw.get("max_tokens", 80),
            ollama_base_url=rewriter_raw.get(
                "ollama_base_url", "http://localhost:11434/v1"
            ),
        )

    return PromptingParams(
        strategy=raw.get("strategy", "vanilla"),
        preset=raw.get("preset"),
        system_prompt=raw.get("system_prompt"),
        prefixes=raw.get("prefixes"),
        suffixes=raw.get("suffixes"),
        rewriter=rewriter_cfg,
    )


# ---------------------------------------------------------------------------
# Fonction principale de chargement
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> RunConfig:
    """
    Charge un fichier YAML et retourne un RunConfig validé.

    Args:
        config_path : chemin vers le fichier YAML (ex: "configs/baseline_groq.yaml")

    Returns:
        RunConfig prêt à l'emploi

    Raises:
        FileNotFoundError si le fichier n'existe pas
        ValueError si la config est invalide
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de config introuvable : {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    logger.info("Chargement de la config : %s", path)

    # Extraction des sections
    gen_raw = raw.get("generation", {})
    prompt_raw = raw.get("prompting", {})
    paths_raw = raw.get("paths", {})

    cfg = RunConfig(
        run_id=raw["run_id"],
        provider=raw["provider"],
        model=raw["model"],
        languages=raw["languages"],
        dataset_type=raw.get("dataset_type", "specific"),
        generation=GenerationParams(
            temperature=gen_raw.get("temperature", 0.0),
            max_tokens=gen_raw.get("max_tokens", 150),
            top_p=gen_raw.get("top_p", 1.0),
        ),
        prompting=_load_prompting(prompt_raw),
        paths=PathsConfig(
            input_dir=Path(paths_raw.get("input_dir", "data/input")),
            output_dir=Path(paths_raw.get("output_dir", "data/output/runs")),
        ),
        # Clé API Groq lue depuis l'environnement (jamais depuis le YAML)
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        ollama_base_url=raw.get(
            "ollama_base_url", "http://localhost:11434/v1"
        ),
        max_questions=raw.get("max_questions"),
        sample_seed=raw.get("sample_seed", 42),
    )

    cfg.validate()
    return cfg
