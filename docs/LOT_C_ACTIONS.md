# Lot C — Actions à exécuter (tests & runs)

Document opérationnel : uniquement les commandes à lancer pour tester et exécuter les runs du Lot C. À dérouler dans l'ordre.

---

## 0. Pré-requis (à vérifier une fois)

```powershell
# Activer l'environnement virtuel
.venv\Scripts\activate

# Vérifier les dépendances
pip install -e ".[dev]"

# Vérifier que GROQ_API_KEY est bien dans .env (variantes C1 et C3)
Get-Content .env | Select-String GROQ_API_KEY

# Lancer Ollama et tirer le modèle Qwen (variantes C2 et C3)
ollama pull qwen2.5:3b
ollama list   # qwen2.5:3b doit apparaître

# Vérifier que les datasets d'entrée sont en place
Get-ChildItem data\input\*.jsonl
# Doit contenir : fr/it/en/es/de × specific/unspecific
```

---

## 1. Tests unitaires (rapide — ~5 sec)

À lancer **avant** tout run pour s'assurer que les 4 stratégies fonctionnent.

```powershell
# Tests des stratégies de prompting (Lot C)
pytest tests/test_prompting.py -v

# Tests complets du projet
pytest tests/ -v
```

**Critère de succès :** tous les tests passent (vanilla + system_prompt + prefix_suffix + rewrite + factory).

---

## 2. Runs de test rapide (5 questions × 5 langues — ~1 min chacun)

Permet de valider chaque variante de bout en bout sans consommer de quota ni attendre.

```powershell
# C1 — System prompt (Groq) 
python run.py --config configs/test_quick_c1_system_prompt.yaml

# C2 — Prefix/Suffix (Qwen local — Ollama doit tourner)
python run.py --config configs/test_quick_c2_prefix_suffix.yaml

# C3 — Rewrite (Qwen local + rewriter Groq)
python run.py --config configs/test_quick_c3_rewrite.yaml
```

**Critère de succès pour chaque test rapide :**
- Le dossier `data/output/runs/test_quick_c{1,2,3}_..._<timestamp>/` est créé
- 5 fichiers `<lang>_specific_output.jsonl` produits (un par langue)
- Chaque ligne JSONL contient bien le champ `prompt_trace` avec la bonne `strategy`
- Pour C3 : `prompt_trace.rewriter_status == "ok"` (et non `fallback_original`)

Vérification rapide d'un fichier de sortie :

```powershell
Get-Content data\output\runs\test_quick_c1_system_prompt_*\fr_specific_output.jsonl | Select-Object -First 1
```

---

## 3. Runs complets des 3 variantes — dataset `specific`

Une fois les tests rapides validés, lancer les runs complets sur les 5 langues × tout le dataset `specific`.

```powershell
# C1 — System prompt sur Groq
python run.py --config configs/variant_c1_system_prompt.yaml

# C2 — Prefix/Suffix sur Qwen local
python run.py --config configs/variant_c2_prefix_suffix.yaml

# C3 — Rewrite (Qwen cible + Groq rewriter)
python run.py --config configs/variant_c3_rewrite.yaml
```

---

## 4. Runs complets des 3 variantes — dataset `unspecific`

Le sujet impose les **deux** types de datasets (robustesse + diversité). Modifier `dataset_type: "specific"` → `"unspecific"` dans chaque YAML, **ou** dupliquer les configs.

Option simple — édition temporaire en place puis relance :

```powershell
# Pour chacune des 3 configs : remplacer "specific" par "unspecific" puis relancer
python run.py --config configs/variant_c1_system_prompt.yaml
python run.py --config configs/variant_c2_prefix_suffix.yaml
python run.py --config configs/variant_c3_rewrite.yaml
```

> Penser à remettre `specific` après le run, ou créer `variant_c{1,2,3}_..._unspecific.yaml` dédiés pour éviter les erreurs.

---

## 5. Vérification finale après les 6 runs (3 variantes × 2 datasets)

```powershell
# Lister tous les dossiers de run du Lot C
Get-ChildItem data\output\runs\variant_c*

# Vérifier qu'on a bien 6 dossiers : c1/c2/c3 × specific/unspecific
# Chaque dossier doit contenir :
#   - config_snapshot.yaml
#   - run_metadata.json
#   - 5 fichiers <lang>_<type>_output.jsonl
```

Contrôles à faire sur chaque `run_metadata.json` :
- `total` correspond bien à la taille du dataset × 5 langues
- `errors` reste faible (< 5 % par langue)
- `avg_latency_ms` cohérent (Groq ~300 ms, Qwen local variable)

---

## 6. Récap — ordre d'exécution résumé

| Ordre | Action | Durée approx. |
|---|---|---|
| 1 | `pytest tests/ -v` | ~5 s |
| 2 | 3× test rapide (C1/C2/C3) | ~3 min total |
| 3 | 3× run complet `specific` | dépend du dataset |
| 4 | 3× run complet `unspecific` | dépend du dataset |
| 5 | Vérification dossiers + `run_metadata.json` | ~2 min |

---

## 7. Astuces de debug

- **Erreur Groq quota / 429** → relancer après quelques minutes, ou réduire `max_questions`.
- **Erreur Ollama connexion refusée** → `ollama serve` dans un terminal à part, vérifier `http://localhost:11434/v1`.
- **Rewriter en `fallback_original` partout (C3)** → vérifier `GROQ_API_KEY`, vérifier les logs côté rewriter.
- **Run trop long pendant le dev** → ajouter `max_questions: 5` + `sample_seed: 42` temporairement dans la config complète.
