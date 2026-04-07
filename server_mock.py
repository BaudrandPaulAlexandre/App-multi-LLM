"""
server_mock.py
--------------
Serveur mock du Lot A — à destination exclusive du Lot B.

Simule exactement les mêmes endpoints que server.py mais retourne
des données factices. Le Lot B peut développer son interface sans
avoir besoin du vrai pipeline, de Groq, d'Ollama ou des fichiers JSONL.

Lancement (Lot B — aucune dépendance Lot A requise) :
    pip install fastapi uvicorn
    uvicorn server_mock:app --reload --port 8001

Le Lot B pointe sur http://localhost:8001 pendant le dev,
puis bascule sur http://localhost:8000 (vrai serveur) pour l'intégration.

Comportement simulé :
    - POST /runs       → retourne un run_id immédiatement
    - GET  /runs/{id}/status → simule une progression sur ~15 secondes
      (avance de 10 questions à chaque appel, puis passe à "done")
    - GET  /runs       → retourne 3 runs d'historique factices
    - GET  /runs/{id}/download → retourne un ZIP factice téléchargeable
    - GET  /providers  → catalogue réel (pas besoin de mocker ça)
"""

from __future__ import annotations

import io
import random
import time
import zipfile
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="ELOQUENT Lot A — Mock Server (développement Lot B)",
    description=(
        "Serveur mock qui simule l'API du Lot A avec des données factices. "
        "Remplacez l'URL de base par http://localhost:8000 pour l'intégration réelle."
    ),
    version="0.1.0-mock",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Catalogue (identique au vrai serveur — pas besoin de mocker)
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

# ---------------------------------------------------------------------------
# État en mémoire des runs simulés
# (dict : run_id → état interne du mock)
# ---------------------------------------------------------------------------

_mock_runs: dict[str, dict] = {}

# Quelques runs d'historique pré-chargés pour que le Lot B ait
# quelque chose à afficher dès le démarrage du mock
_mock_runs["groq_vanilla_20260320_090000"] = {
    "run_id": "groq_vanilla_20260320_090000",
    "status": "done",
    "provider": "groq",
    "model": "llama-3.1-8b-instant",
    "strategy": "vanilla",
    "dataset_type": "specific",
    "languages": ["fr", "it", "en", "es", "de"],
    "started_at": "2026-03-20T09:00:00Z",
    "ended_at": "2026-03-20T09:08:12Z",
    "duration_seconds": 492.0,
    "per_language": {
        lang: {"total": 100, "success": 98, "errors": 2, "avg_latency_ms": 410}
        for lang in ["fr", "it", "en", "es", "de"]
    },
}
_mock_runs["qwen_ollama_vanilla_20260321_143000"] = {
    "run_id": "qwen_ollama_vanilla_20260321_143000",
    "status": "done",
    "provider": "qwen_ollama",
    "model": "qwen2.5:3b",
    "strategy": "vanilla",
    "dataset_type": "unspecific",
    "languages": ["fr", "en"],
    "started_at": "2026-03-21T14:30:00Z",
    "ended_at": "2026-03-21T14:52:44Z",
    "duration_seconds": 1364.0,
    "per_language": {
        "fr": {"total": 100, "success": 100, "errors": 0, "avg_latency_ms": 820},
        "en": {"total": 100, "success": 99,  "errors": 1, "avg_latency_ms": 790},
    },
}
_mock_runs["groq_vanilla_20260322_110000"] = {
    "run_id": "groq_vanilla_20260322_110000",
    "status": "error",
    "provider": "groq",
    "model": "llama-3.3-70b-versatile",
    "strategy": "vanilla",
    "dataset_type": "specific",
    "languages": ["fr"],
    "started_at": "2026-03-22T11:00:00Z",
    "duration_seconds": None,
    "error": "GROQ_API_KEY invalide — vérifiez votre fichier .env",
}


# ---------------------------------------------------------------------------
# Schéma de requête (identique au vrai serveur)
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    provider: str = "groq"
    model: str = "llama-3.1-8b-instant"
    languages: list[str] = ["fr", "it", "en", "es", "de"]
    dataset_type: str = "specific"
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(150, ge=10, le=2048)
    strategy: str = "vanilla"


# ---------------------------------------------------------------------------
# Endpoints (surface identique au vrai serveur)
# ---------------------------------------------------------------------------

@app.get("/providers", tags=["Catalogue"])
def get_providers() -> dict:
    return {
        "providers":  AVAILABLE_PROVIDERS,
        "languages":  AVAILABLE_LANGUAGES,
        "strategies": AVAILABLE_STRATEGIES,
    }


@app.post("/runs", status_code=202, tags=["Runs"])
def create_run(req: RunRequest) -> dict:
    """
    Lance un run simulé. La progression avancera automatiquement
    de 10 questions à chaque appel à GET /runs/{run_id}/status.
    Le run passe à "done" une fois toutes les questions traitées.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{req.provider}_{req.strategy}_{timestamp}"

    total_questions = len(req.languages) * 100   # 100 questions simulées par langue

    _mock_runs[run_id] = {
        "run_id":           run_id,
        "status":           "running",
        "provider":         req.provider,
        "model":            req.model,
        "strategy":         req.strategy,
        "dataset_type":     req.dataset_type,
        "languages":        req.languages,
        "started_at":       datetime.now(timezone.utc).isoformat(),
        "generation": {
            "temperature": req.temperature,
            "max_tokens":  req.max_tokens,
        },
        # Champs internes du mock (pas exposés dans /status)
        "_questions_done":  0,
        "_questions_total": total_questions,
        "_errors":          0,
        "_languages_done":  [],
        "_current_lang":    req.languages[0] if req.languages else "",
    }

    return {
        "run_id":  run_id,
        "run_dir": f"/mock/data/output/runs/{run_id}",
        "status":  "started",
        "message": f"[MOCK] Run '{run_id}' simulé lancé.",
    }


@app.get("/runs", tags=["Runs"])
def list_runs() -> list[dict]:
    """Retourne l'historique des runs (vrais + simulés)."""
    result = []
    for run in sorted(_mock_runs.values(), key=lambda r: r.get("started_at", ""), reverse=True):
        result.append({
            "run_id":           run["run_id"],
            "run_dir":          run.get("run_dir", f"/mock/{run['run_id']}"),
            "status":           run["status"],
            "provider":         run.get("provider", "?"),
            "model":            run.get("model", "?"),
            "strategy":         run.get("strategy", "?"),
            "dataset_type":     run.get("dataset_type", "?"),
            "languages":        run.get("languages", []),
            "started_at":       run.get("started_at", ""),
            "duration_seconds": run.get("duration_seconds"),
        })
    return result


@app.get("/runs/{run_id}/status", tags=["Runs"])
def get_run_status(run_id: str) -> dict:
    """
    Retourne la progression simulée du run.
    Avance de 10 questions à chaque appel — simule le travail du pipeline.
    Passe automatiquement à "done" quand toutes les questions sont traitées.
    """
    if run_id not in _mock_runs:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' introuvable.")

    run = _mock_runs[run_id]

    # Si le run est déjà terminé, retourner l'état final directement
    if run["status"] in ("done", "error"):
        return _public_status(run)

    # Simuler l'avancement : +10 questions à chaque appel de status
    done = run["_questions_done"]
    total = run["_questions_total"]
    step = 10

    # Simuler une erreur aléatoire (5% de chance)
    if random.random() < 0.05:
        run["_errors"] += 1
        run["_questions_done"] = min(done + step - 1, total)
    else:
        run["_questions_done"] = min(done + step, total)

    done = run["_questions_done"]

    # Mettre à jour la langue courante simulée
    langs = run["languages"]
    questions_per_lang = total // len(langs) if langs else 1
    current_lang_idx = min(done // questions_per_lang, len(langs) - 1)
    run["_current_lang"] = langs[current_lang_idx]
    run["_languages_done"] = langs[:current_lang_idx]

    # Run terminé ?
    if done >= total:
        run["status"] = "done"
        run["ended_at"] = datetime.now(timezone.utc).isoformat()
        run["duration_seconds"] = round(
            (datetime.now(timezone.utc) -
             datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))
            ).total_seconds(), 1
        )
        run["per_language"] = {
            lang: {
                "total":          100,
                "success":        100 - (1 if lang == langs[0] else 0),
                "errors":         1 if lang == langs[0] else 0,
                "avg_latency_ms": random.randint(380, 520),
            }
            for lang in langs
        }

    return _public_status(run)


@app.get("/runs/{run_id}/download", tags=["Runs"])
def download_run(run_id: str) -> StreamingResponse:
    """
    Retourne un ZIP factice contenant des fichiers JSONL simulés.
    Permet au Lot B de tester son bouton d'export sans attendre un vrai run.
    """
    if run_id not in _mock_runs:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' introuvable.")

    run = _mock_runs[run_id]
    if run["status"] not in ("done", "error"):
        raise HTTPException(status_code=409, detail="Run encore en cours.")

    langs = run.get("languages", ["fr"])

    # Construction du ZIP en mémoire avec des données factices
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:

        # config_snapshot.yaml
        zf.writestr("config_snapshot.yaml",
            f"run_id: {run_id}\nprovider: {run['provider']}\nmodel: {run['model']}\n"
        )

        # run_metadata.json
        import json
        zf.writestr("run_metadata.json", json.dumps({
            "run_id": run_id,
            "status": run["status"],
            "provider": run["provider"],
            "model": run["model"],
        }, ensure_ascii=False, indent=2))

        # Un fichier JSONL par langue avec 3 exemples
        for lang in langs:
            lines = []
            sample_answers = {
                "fr": "La réponse est disponible dans les documents officiels.",
                "it": "La risposta è disponibile nei documenti ufficiali.",
                "en": "The answer is available in the official documents.",
                "es": "La respuesta está disponible en los documentos oficiales.",
                "de": "Die Antwort ist in den offiziellen Dokumenten verfügbar.",
            }
            for i in range(3):
                lines.append(json.dumps({
                    "id": f"{lang}_{i+1:03d}",
                    "question": f"[MOCK] Question {i+1} en {lang}",
                    "answer": sample_answers.get(lang, "Mock answer."),
                }, ensure_ascii=False))
            zf.writestr(
                f"{lang}_{run['dataset_type']}_output.jsonl",
                "\n".join(lines) + "\n",
            )

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={run_id}.zip"},
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _public_status(run: dict) -> dict:
    """Filtre les champs internes du mock avant de répondre au client."""
    return {k: v for k, v in run.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Lancement direct
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("\n🟡 Mock server démarré sur http://localhost:8001")
    print("📖 Documentation : http://localhost:8001/docs\n")
    uvicorn.run("server_mock:app", host="0.0.0.0", port=8001, reload=True)
