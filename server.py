"""
server.py
---------
Serveur FastAPI — point d'entrée du Lot A côté réseau.

Le Lot B (et tout autre client) communique UNIQUEMENT via ce serveur.
Aucun import direct de pipeline.py ou providers.py n'est nécessaire côté client.

Lancement :
    uvicorn server:app --reload --host 0.0.0.0 --port 8000

Documentation interactive auto-générée :
    http://localhost:8000/docs

Endpoints exposés :
    GET  /providers              → liste providers + modèles
    POST /runs                   → lancer un run (retour immédiat)
    GET  /runs                   → historique des runs
    GET  /runs/{run_id}/status   → progression en temps réel
    GET  /runs/{run_id}/download → télécharger le package ZIP
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from eloquent.config import (
    GenerationParams, PathsConfig, PromptingParams, RunConfig,
)
from eloquent.logger import get_logger, setup_logging
from eloquent.pipeline import PipelineRunner
from eloquent.providers import build_provider_from_config, test_determinism

setup_logging()
logger = get_logger("server")

# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ELOQUENT Lot A — Pipeline API",
    description=(
        "API REST du pipeline backend multi-LLM pour le challenge "
        "ELOQUENT CLEF 2026. Consommée par le Lot B (interface utilisateur)."
    ),
    version="0.1.0",
)

# CORS : autorise le Lot B à appeler l'API depuis n'importe quelle origine
# En production, remplacer ["*"] par l'URL exacte du frontend Lot B
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constantes (catalogue providers/langues/stratégies)
# ---------------------------------------------------------------------------

AVAILABLE_PROVIDERS = {
    "groq": {
        "label": "Groq API (cloud)",
        "requires_api_key": True,
        "models": [
            {"id": "llama-3.1-8b-instant",    "label": "Llama 3.1 8B  (rapide)"},
            {"id": "llama-3.3-70b-versatile", "label": "Llama 3.3 70B (puissant)"},
            {"id": "mixtral-8x7b-32768",       "label": "Mixtral 8x7B"},
        ],
    },
    "qwen_ollama": {
        "label": "Qwen 2.5 3B — local (Ollama)",
        "requires_api_key": False,
        "models": [
            {"id": "qwen2.5:3b", "label": "Qwen 2.5 3B (local)"},
        ],
    },
}

AVAILABLE_LANGUAGES = [
    {"code": "fr", "label": "Français"},
    {"code": "it", "label": "Italien"},
    {"code": "en", "label": "Anglais"},
    {"code": "es", "label": "Espagnol"},
    {"code": "de", "label": "Allemand"},
]

AVAILABLE_STRATEGIES = [
    {"id": "vanilla",       "label": "Vanilla — texte brut (baseline)"},
    {"id": "system_prompt", "label": "System prompt (Lot C)"},
]

OUTPUT_DIR = Path("data/output/runs")
INPUT_DIR  = Path("data/input")

# ---------------------------------------------------------------------------
# Modèles Pydantic (schémas de requête/réponse — générés dans /docs)
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    """Corps de la requête POST /runs — ce que le Lot B envoie."""
    provider: str = Field(
        "groq",
        description="Identifiant du provider : 'groq' ou 'qwen_ollama'",
    )
    model: str = Field(
        "llama-3.1-8b-instant",
        description="Identifiant du modèle (voir GET /providers)",
    )
    languages: list[str] = Field(
        ["fr", "it", "en", "es", "de"],
        description="Codes langue à traiter",
    )
    dataset_type: str = Field(
        "specific",
        description="'specific' ou 'unspecific'",
    )
    temperature: float = Field(
        0.0,
        ge=0.0, le=2.0,
        description="0.0 = déterministe (obligatoire pour la baseline)",
    )
    max_tokens: int = Field(
        150,
        ge=10, le=2048,
        description="Longueur max de la réponse (~150 pour ELOQUENT)",
    )
    strategy: str = Field(
        "vanilla",
        description="Stratégie de prompting : 'vanilla' ou 'system_prompt'",
    )


class RunSummary(BaseModel):
    """Résumé d'un run retourné par GET /runs et POST /runs."""
    run_id: str
    run_dir: str
    status: str
    provider: str
    model: str
    strategy: str
    dataset_type: str
    languages: list[str]
    started_at: str
    duration_seconds: float | None = None


# ---------------------------------------------------------------------------
# Registre en mémoire des threads de run actifs
# (permet de savoir si un run est encore en cours)
# ---------------------------------------------------------------------------

_active_runs: dict[str, threading.Thread] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/providers",
    summary="Liste les providers et modèles disponibles",
    tags=["Catalogue"],
)
def get_providers() -> dict:
    """
    Retourne le catalogue complet des providers, modèles, langues
    et stratégies disponibles.

    Le Lot B utilise cette réponse pour peupler tous ses menus déroulants
    sans avoir à coder les valeurs en dur.
    """
    return {
        "providers":  AVAILABLE_PROVIDERS,
        "languages":  AVAILABLE_LANGUAGES,
        "strategies": AVAILABLE_STRATEGIES,
    }


@app.post(
    "/runs",
    summary="Lance un nouveau run",
    status_code=202,   # 202 Accepted = traitement en arrière-plan
    tags=["Runs"],
)
def create_run(req: RunRequest) -> dict:
    """
    Lance un run ELOQUENT en arrière-plan et retourne immédiatement
    le run_id et le chemin du dossier de run.

    Le Lot B peut ensuite poller GET /runs/{run_id}/status pour
    suivre la progression.

    Retourne HTTP 202 (Accepted) et non 200 (OK) car le traitement
    est asynchrone — la réponse arrive avant la fin du run.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id    = f"{req.provider}_{req.strategy}_{timestamp}"

    cfg = RunConfig(
        run_id=run_id,
        provider=req.provider,
        model=req.model,
        languages=req.languages,
        dataset_type=req.dataset_type,
        generation=GenerationParams(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        ),
        prompting=PromptingParams(strategy=req.strategy),
        paths=PathsConfig(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
        ),
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )

    try:
        cfg.validate()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    runner = PipelineRunner(cfg)

    def _run():
        try:
            runner.run()
        except Exception as exc:
            logger.error("Run '%s' échoué : %s", run_id, exc)

    thread = threading.Thread(target=_run, daemon=True, name=run_id)
    thread.start()
    _active_runs[run_id] = thread

    logger.info("Run lancé : %s", run_id)

    return {
        "run_id":  run_id,
        "run_dir": str(runner.run_dir),
        "status":  "started",
        "message": f"Run '{run_id}' lancé. Pollez GET /runs/{run_id}/status.",
    }


@app.get(
    "/runs",
    summary="Liste l'historique des runs",
    tags=["Runs"],
)
def list_runs() -> list[dict]:
    """
    Retourne la liste de tous les runs passés et en cours,
    triée du plus récent au plus ancien.

    Utilisé par le Lot B pour afficher l'historique dans son interface.
    """
    if not OUTPUT_DIR.exists():
        return []

    runs = []
    for run_dir in sorted(OUTPUT_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        status_data = _read_run_status(run_dir)
        runs.append(status_data)

    return runs


@app.get(
    "/runs/{run_id}/status",
    summary="Progression en temps réel d'un run",
    tags=["Runs"],
)
def get_run_status(run_id: str) -> dict:
    """
    Retourne l'état courant du run : progression question par question,
    langue en cours, erreurs.

    Le Lot B appelle cet endpoint toutes les ~2 secondes pour mettre
    à jour sa barre de progression.

    Champ 'status' possible :
      - "pending"  : run créé mais pas encore démarré
      - "running"  : en cours de traitement
      - "done"     : terminé avec succès
      - "error"    : terminé avec erreur critique
    """
    # Cherche le dossier correspondant au run_id
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' introuvable dans {OUTPUT_DIR}",
        )
    return _read_run_status(run_dir)


@app.get(
    "/runs/{run_id}/download",
    summary="Télécharger le package ZIP d'un run",
    tags=["Runs"],
)
def download_run(run_id: str) -> StreamingResponse:
    """
    Génère et retourne un fichier ZIP contenant le package complet du run :
      - config_snapshot.yaml
      - run_metadata.json
      - fr_specific_output.jsonl
      - it_specific_output.jsonl
      - ... (un fichier par langue)

    Ce ZIP est le "package de soumission" au format ELOQUENT.
    Le Lot B n'a qu'à exposer un bouton qui déclenche ce téléchargement.
    """
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' introuvable",
        )

    # Vérification que le run est terminé
    status_data = _read_run_status(run_dir)
    if status_data.get("status") not in ("done", "error"):
        raise HTTPException(
            status_code=409,    # Conflict
            detail="Le run est encore en cours. Attendez qu'il soit terminé.",
        )

    # Construction du ZIP en mémoire (pas de fichier temporaire sur disque)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(run_dir.iterdir()):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)
    zip_buffer.seek(0)

    filename = f"{run_id}.zip"
    logger.info("Téléchargement package : %s", filename)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _find_run_dir(run_id: str) -> Path | None:
    """Trouve le dossier d'un run par son run_id (préfixe du nom de dossier)."""
    if not OUTPUT_DIR.exists():
        return None
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir() and d.name.startswith(run_id):
            return d
    return None


def _read_run_status(run_dir: Path) -> dict:
    """
    Lit progress.json (run en cours) ou run_metadata.json (run terminé).
    Retourne toujours un dict avec au minimum {"status": ..., "run_id": ...}.
    """
    metadata_file = run_dir / "run_metadata.json"
    progress_file = run_dir / "progress.json"

    if metadata_file.exists():
        data = json.loads(metadata_file.read_text(encoding="utf-8"))
        data["status"] = "done"
        data["run_dir"] = str(run_dir)
        return data

    if progress_file.exists():
        data = json.loads(progress_file.read_text(encoding="utf-8"))
        data["run_dir"] = str(run_dir)
        return data

    return {
        "run_id":  run_dir.name,
        "run_dir": str(run_dir),
        "status":  "pending",
    }


# ---------------------------------------------------------------------------
# Lancement direct (développement)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
