"""
run.py
------
Point d'entrée CLI du pipeline ELOQUENT Lot A.

Usage :
    python run.py --config configs/baseline_groq.yaml
    python run.py --config configs/baseline_qwen.yaml
    python run.py --config configs/baseline_groq.yaml --skip-determinism-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# -- Permet d'importer le package eloquent sans pip install --
sys.path.insert(0, str(Path(__file__).parent / "src"))

from eloquent.config import load_config
from eloquent.logger import get_logger, setup_logging
from eloquent.pipeline import PipelineRunner
from eloquent.providers import build_provider_from_config, test_determinism


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline ELOQUENT — Lot A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python run.py --config configs/baseline_groq.yaml
  python run.py --config configs/baseline_qwen.yaml --skip-determinism-check
        """,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Chemin vers le fichier de configuration YAML",
    )
    parser.add_argument(
        "--skip-determinism-check",
        action="store_true",
        default=False,
        help="Ignore le test de déterminisme avant le run (non recommandé pour la baseline)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de log (défaut : INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Chargement de la config
    cfg = load_config(args.config)

    # 2. Configuration du logging
    #    Les logs sont écrits dans la console ET dans un fichier dans le run dir
    import logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)
    logger = get_logger("run")

    logger.info("=" * 60)
    logger.info("ELOQUENT Pipeline — Lot A")
    logger.info("Config : %s", args.config)
    logger.info("Run ID : %s", cfg.run_id)
    logger.info("Provider : %s | Modèle : %s", cfg.provider, cfg.model)
    logger.info("Langues : %s", cfg.languages)
    logger.info("Dataset : %s", cfg.dataset_type)
    logger.info("=" * 60)

    # 3. Test de déterminisme (obligatoire pour la baseline sauf si --skip)
    if not args.skip_determinism_check:
        logger.info("Test de déterminisme (requis pour la baseline)...")
        provider = build_provider_from_config(cfg)
        is_det = test_determinism(provider, n_runs=2)
        if not is_det:
            logger.warning(
                "⚠️  Le provider n'est pas strictement déterministe. "
                "Les résultats de la baseline peuvent varier légèrement. "
                "À documenter dans le rapport."
            )
    else:
        logger.info("Test de déterminisme ignoré (--skip-determinism-check).")

    # 4. Lancement du pipeline
    runner = PipelineRunner(cfg)
    metadata = runner.run()

    # 5. Résumé final
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ Run terminé : %s", cfg.run_id)
    logger.info("Durée totale : %.1fs", metadata["duration_seconds"])
    for lang, stats in metadata["per_language"].items():
        if stats.get("skipped"):
            logger.info("  [%s] ignoré (fichier introuvable)", lang)
        else:
            logger.info(
                "  [%s] %d questions | %d OK | %d erreurs | moy. %.0fms",
                lang,
                stats["total"],
                stats["success"],
                stats["errors"],
                stats["avg_latency_ms"],
            )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
