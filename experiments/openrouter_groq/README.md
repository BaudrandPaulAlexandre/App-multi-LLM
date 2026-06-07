# Variante baseline — Passerelle OpenRouter → Groq

> **Dossier jetable.** Tout ce qui concerne cette variante vit ici.
> Pour tout retirer : **supprimez ce dossier** (`experiments/openrouter_groq/`).
> Aucun fichier source du package `eloquent` n'a été modifié.

## Principe

Au lieu d'appeler l'API Groq directement (comme `configs/baseline_groq.yaml`),
on passe par **OpenRouter** comme passerelle, en **forçant** OpenRouter à router
l'inférence vers **Groq** via son *provider routing* :

```json
{ "provider": { "order": ["Groq"], "allow_fallbacks": false } }
```

`allow_fallbacks: false` garantit que si Groq est indisponible, l'appel
**échoue** plutôt que de basculer silencieusement sur un autre fournisseur —
indispensable pour que la comparaison avec la baseline Groq directe reste valide.

Le modèle utilisé est l'équivalent OpenRouter du `llama-3.1-8b-instant` :
`meta-llama/llama-3.1-8b-instruct`.

## Prérequis

1. Une clé OpenRouter dans le `.env` à la racine du projet :
   ```
   OPENROUTER_API_KEY=sk-or-...
   ```
2. Le paquet `openai` (déjà installé — utilisé par le provider Qwen).

## Lancer

```powershell
python experiments/openrouter_groq/run.py
# ou en sautant le test de déterminisme :
python experiments/openrouter_groq/run.py --skip-determinism-check
```

Les résultats sont écrits dans `data/output/runs/baseline_openrouter_groq_vanilla_<timestamp>/`
— exactement au même format que les autres baselines, donc directement
comparables dans les notebooks d'analyse.

## Comment ça reste isolé

`run.py` réutilise le `PipelineRunner` du package sans le modifier :
il construit le `RunConfig`, instancie le runner, puis **remplace**
`runner.provider` par notre `OpenRouterGroqProvider` avant de lancer.
La validation de `config.py` ne connaît pas `openrouter_groq`, donc on
déclare temporairement `provider="groq"` dans le `RunConfig` (placeholder)
juste pour passer la validation.

## Fichiers

| Fichier        | Rôle                                                        |
|----------------|-------------------------------------------------------------|
| `config.yaml`  | Config de la variante (modèle, routing, langues, génération)|
| `provider.py`  | `OpenRouterGroqProvider` (API OpenAI-compatible + routing)  |
| `run.py`       | Runner autonome qui injecte le provider dans le pipeline    |
