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
class PromptingParams:
    strategy: str = "vanilla"   # "vanilla" | "system_prompt" | "rewrite" (Lot C)


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

    max_questions: int | None = None   
    sample_seed: int = 42   

    # Champs optionnels selon le provider
    groq_api_key: str | None = None       # Lu depuis .env si provider=groq
    ollama_base_url: str = "http://localhost:11434/v1"

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

        logger.info("Config '%s' validée ✓", self.run_id)

    def to_dict(self) -> dict:
        """Sérialise la config en dict (pour la sauvegarder dans le run)."""
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "languages": self.languages,
            "dataset_type": self.dataset_type,
            "generation": self.generation.to_dict(),
            "prompting": {"strategy": self.prompting.strategy},
            "ollama_base_url": self.ollama_base_url,
            "max_questions": self.max_questions,
            "sample_seed":   self.sample_seed,
            # On ne sérialise JAMAIS la clé API
        }


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
        max_questions=raw.get("max_questions", None),
        sample_seed=raw.get("sample_seed", 42),
        generation=GenerationParams(
            temperature=gen_raw.get("temperature", 0.0),
            max_tokens=gen_raw.get("max_tokens", 150),
            top_p=gen_raw.get("top_p", 1.0),
        ),
        prompting=PromptingParams(
            strategy=prompt_raw.get("strategy", "vanilla"),
        ),
        paths=PathsConfig(
            input_dir=Path(paths_raw.get("input_dir", "data/input")),
            output_dir=Path(paths_raw.get("output_dir", "data/output/runs")),
        ),
        # Clé API Groq lue depuis l'environnement (jamais depuis le YAML)
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        ollama_base_url=raw.get(
            "ollama_base_url", "http://localhost:11434/v1"
        ),
    )

    cfg.validate()
    return cfg
