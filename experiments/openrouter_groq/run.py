"""
run.py — Runner autonome pour la variante OpenRouter -> Groq
===========================================================

Réutilise le PipelineRunner du package eloquent SANS le modifier :
on construit le RunConfig, on instancie le PipelineRunner, puis on
remplace son `.provider` par notre OpenRouterGroqProvider.

Pourquoi ce contournement ?
  config.py n'accepte que les providers {"groq", "qwen_ollama"} et lèverait
  une erreur sur "openrouter_groq". Pour rester 100% non-intrusif (dossier
  jetable), on déclare provider="groq" dans le RunConfig (juste pour passer
  la validation + lire GROQ_API_KEY sans qu'elle soit requise), puis on
  écrase le provider construit par la factory avec le nôtre.

Usage :
    python experiments/openrouter_groq/run.py
    python experiments/openrouter_groq/run.py --config experiments/openrouter_groq/config.yaml
    python experiments/openrouter_groq/run.py --skip-determinism-check

Prérequis :
    - OPENROUTER_API_KEY dans le .env (à la racine du projet)
    - pip install openai  (déjà installé : utilisé par le provider Qwen)

>>> DOSSIER JETABLE : supprimer experiments/openrouter_groq/ retire tout. <<<
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# -- Racine du projet : permet d'importer le package eloquent sans pip install --
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # pour importer provider.py

from eloquent.config import (  # noqa: E402
    GenerationParams,
    PathsConfig,
    PromptingParams,
    RunConfig,
)
from eloquent.logger import get_logger, setup_logging  # noqa: E402
from eloquent.pipeline import PipelineRunner  # noqa: E402

from provider import OpenRouterGroqProvider  # noqa: E402

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Variante baseline ELOQUENT — passerelle OpenRouter -> Groq",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Chemin vers le YAML (défaut : {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--skip-determinism-check",
        action="store_true",
        default=False,
        help="Ignore le test de déterminisme avant le run",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def build_runconfig(raw: dict) -> RunConfig:
    """
    Construit un RunConfig depuis le YAML de cette variante.

    Astuce d'isolation : on déclare provider="groq" pour que la validation
    de config.py passe (elle ne connaît pas "openrouter_groq"). Le vrai
    provider sera injecté juste après dans le PipelineRunner.
    """
    gen_raw = raw.get("generation", {})
    paths_raw = raw.get("paths", {})

    cfg = RunConfig(
        run_id=raw["run_id"],
        provider="groq",  # placeholder pour la validation — écrasé plus bas
        model=raw["model"],
        languages=raw["languages"],
        dataset_type=raw.get("dataset_type", "specific"),
        generation=GenerationParams(
            temperature=gen_raw.get("temperature", 0.0),
            max_tokens=gen_raw.get("max_tokens", 150),
            top_p=gen_raw.get("top_p", 1.0),
        ),
        prompting=PromptingParams(strategy=raw.get("prompting", {}).get("strategy", "vanilla")),
        paths=PathsConfig(
            input_dir=Path(paths_raw.get("input_dir", "data/input")),
            output_dir=Path(paths_raw.get("output_dir", "data/output/runs")),
        ),
        # validate() exige groq_api_key quand provider=="groq". On lit GROQ_API_KEY
        # si présente ; sinon on met un placeholder (la clé Groq n'est PAS utilisée,
        # c'est OPENROUTER_API_KEY qui sert réellement).
        groq_api_key=os.environ.get("GROQ_API_KEY") or "unused-openrouter-gateway",
        max_questions=raw.get("max_questions"),
        sample_seed=raw.get("sample_seed", 42),
    )
    cfg.validate()
    return cfg


def main() -> None:
    args = parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    setup_logging(level=getattr(logging, args.log_level))
    logger = get_logger("run_openrouter_groq")

    or_cfg = raw.get("openrouter", {})
    base_url = or_cfg.get("base_url", OpenRouterGroqProvider.DEFAULT_BASE_URL)
    provider_order = or_cfg.get("provider_order", ["Groq"])
    allow_fallbacks = or_cfg.get("allow_fallbacks", False)
    request_delay_s = or_cfg.get("request_delay_s", 0.0)

    logger.info("=" * 60)
    logger.info("ELOQUENT — Variante OpenRouter -> Groq")
    logger.info("Config : %s", args.config)
    logger.info("Run ID : %s", raw["run_id"])
    logger.info("Modèle : %s", raw["model"])
    logger.info("Routing : order=%s | allow_fallbacks=%s", provider_order, allow_fallbacks)
    logger.info("Délai inter-question : %.2fs", request_delay_s)
    logger.info("Langues : %s", raw["languages"])
    logger.info("=" * 60)

    cfg = build_runconfig(raw)

    # --- Le provider OpenRouter -> Groq, construit ici ---
    provider = OpenRouterGroqProvider(
        model=raw["model"],
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url=base_url,
        provider_order=provider_order,
        allow_fallbacks=allow_fallbacks,
        request_delay_s=request_delay_s,
    )

    # Test de déterminisme (optionnel)
    if not args.skip_determinism_check:
        from eloquent.providers import test_determinism
        logger.info("Test de déterminisme...")
        if not test_determinism(provider, n_runs=2):
            logger.warning(
                "⚠️  Provider non strictement déterministe. "
                "À documenter dans le rapport."
            )
    else:
        logger.info("Test de déterminisme ignoré (--skip-determinism-check).")

    # --- Injection du provider dans le PipelineRunner existant ---
    runner = PipelineRunner(cfg)
    runner.provider = provider  # on écrase le GroqProvider de la factory

    metadata = runner.run()

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ Run terminé : %s", cfg.run_id)
    logger.info("Durée totale : %.1fs", metadata["duration_seconds"])
    for lang, stats in metadata["per_language"].items():
        if stats.get("skipped"):
            logger.info("  [%s] ignoré (fichier introuvable)", lang)
        else:
            logger.info(
                "  [%s] %d/%d questions | %d OK | %d erreurs | moy. %.0fms",
                lang,
                stats["total_sampled"],
                stats["total_in_file"],
                stats["success"],
                stats["errors"],
                stats["avg_latency_ms"],
            )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
