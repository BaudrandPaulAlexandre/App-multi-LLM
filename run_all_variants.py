"""
run_all_variants.py
-------------------
Lance les 3 variantes du Lot C (C1 / C2 / C3) à la suite, avec un mode
test FR-only par défaut (configs/test_fr_c{1,2,3}_*.yaml).

Usage :
    # Mode test rapide FR-only (par défaut)
    python run_all_variants.py

    # Mode complet (5 langues × dataset complet)
    python run_all_variants.py --full

    # Choisir explicitement les configs à enchaîner
    python run_all_variants.py --configs configs/a.yaml configs/b.yaml

Chaque run est isolé : un échec sur une variante n'arrête pas les suivantes.
Un récapitulatif des codes de sortie est imprimé à la fin.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DEFAULT_TEST_CONFIGS = [
    "configs/test_fr_c1_system_prompt.yaml",
    "configs/test_fr_c2_prefix_suffix.yaml",
    "configs/test_fr_c3_rewrite.yaml",
]

FULL_CONFIGS = [
    "configs/variant_c1_system_prompt.yaml",
    "configs/variant_c2_prefix_suffix.yaml",
    "configs/variant_c3_rewrite.yaml",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lance les variantes Lot C en chaîne.")
    p.add_argument(
        "--full",
        action="store_true",
        help="Utilise les configs complètes (5 langues) au lieu des tests FR-only.",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        metavar="PATH",
        help="Chemins de configs à enchaîner (override --full / défaut).",
    )
    return p.parse_args()


def run_one(config_path: str) -> tuple[int, float]:
    """Lance `python run.py --config <config_path>` et renvoie (returncode, durée_sec)."""
    print(f"\n{'=' * 70}\n>>> RUN : {config_path}\n{'=' * 70}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "run.py", "--config", config_path],
        cwd=ROOT,
    )
    duration = time.perf_counter() - started
    return result.returncode, duration


def main() -> int:
    args = parse_args()

    if args.configs:
        configs = args.configs
    elif args.full:
        configs = FULL_CONFIGS
    else:
        configs = DEFAULT_TEST_CONFIGS

    # Vérification d'existence avant de lancer quoi que ce soit
    missing = [c for c in configs if not (ROOT / c).is_file()]
    if missing:
        print("ERREUR — configs introuvables :", file=sys.stderr)
        for c in missing:
            print(f"  - {c}", file=sys.stderr)
        return 2

    print(f"Mode : {'FULL' if args.full else 'TEST FR-only'}")
    print(f"Configs à exécuter ({len(configs)}) :")
    for c in configs:
        print(f"  - {c}")

    results: list[tuple[str, int, float]] = []
    for cfg in configs:
        rc, dur = run_one(cfg)
        results.append((cfg, rc, dur))

    # Récapitulatif final
    print(f"\n{'=' * 70}\nRECAP\n{'=' * 70}")
    total = 0.0
    failures = 0
    for cfg, rc, dur in results:
        status = "OK " if rc == 0 else f"FAIL (rc={rc})"
        print(f"  [{status}] {cfg}  ({dur:.1f}s)")
        total += dur
        if rc != 0:
            failures += 1
    print(f"\nTotal : {total:.1f}s — {len(results) - failures}/{len(results)} OK")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
