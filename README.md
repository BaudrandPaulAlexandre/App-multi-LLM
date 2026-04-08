# ELOQUENT Lot A — Pipeline Backend Multi-LLM

Pipeline backend pour le challenge **ELOQUENT Cultural Robustness & Diversity**
(CLEF 2026). Ce dépôt correspond au **Lot A** : exécution de runs et gestion
de plusieurs modèles LLM.

## Prérequis

- Python 3.11+
- [Ollama](https://ollama.com/download) (pour le modèle Qwen local)

## Installation

```bash
# 1. Cloner le dépôt
git clone <url-du-repo>
cd eloquent-lot-a

# 2. Créer un environnement virtuel
python -m venv .venv
# source .venv/bin/activate          # Linux / Mac
 .venv\Scripts\activate           # Windows
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser # En cas de problème avec Windows

# 3. Installer les dépendances
pip install -e ".[dev]"

# 4. Configurer les clés API
cp .env.example .env
# Éditez .env et renseignez votre GROQ_API_KEY
```

## Données ELOQUENT

Téléchargez les fichiers JSONL depuis :
https://github.com/eloquent-lab/

Placez-les dans `data/input/` avec la nomenclature :
```
data/input/fr_specific.jsonl
data/input/fr_unspecific.jsonl
data/input/it_specific.jsonl
... (un fichier par langue × type)
```

## Lancer un run

```bash
# Baseline Groq (llama-3.1-8b-instant, temperature=0)
python run.py --config configs/baseline_groq.yaml

# Baseline Qwen local (Ollama doit tourner)
ollama pull qwen2.5:3b
python run.py --config configs/baseline_qwen.yaml

# Sans test de déterminisme (plus rapide)
python run.py --config configs/baseline_groq.yaml --skip-determinism-check
```

Les résultats sont écrits dans `data/output/runs/<run_id>_<timestamp>/`.

## Lancer les tests

```bash
pytest tests/ -v
```

## Structure du projet

```
src/eloquent/
    providers.py   — LLMProvider, GroqProvider, QwenOllamaProvider
    pipeline.py    — PipelineRunner (boucle principale)
    config.py      — Chargement et validation du YAML
    prompting.py   — Stratégies de prompting (vanilla, system_prompt)
    logger.py      — Configuration du logging

configs/
    baseline_groq.yaml   — Baseline Groq, temperature=0
    baseline_qwen.yaml   — Baseline Qwen local, temperature=0

run.py             — Point d'entrée CLI
```

## Démarrer le serveur (API REST pour le Lot B)

```bash
# Installer les dépendances (FastAPI + uvicorn ajoutés)
pip install -e ".[dev]"

# Lancer le serveur
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# Documentation interactive auto-générée :
# http://localhost:8000/docs
```

## Communication avec le Lot B

Le Lot B interagit via HTTP — aucun import Python direct.

```
POST http://localhost:8000/runs          ← lancer un run
GET  http://localhost:8000/runs          ← historique
GET  http://localhost:8000/runs/{id}/status   ← progression
GET  http://localhost:8000/runs/{id}/download ← package ZIP
GET  http://localhost:8000/providers     ← catalogue providers/modèles
```

Pour tester sans le Lot B :
```bash
curl http://localhost:8000/providers
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"provider":"groq","model":"llama-3.1-8b-instant","languages":["fr"],"dataset_type":"specific"}'
```
