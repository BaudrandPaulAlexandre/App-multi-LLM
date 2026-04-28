# Lot C — Variantes : prompting et reformulation automatique

**Projet** : Application multi-LLM pour évaluer la robustesse culturelle et la diversité des réponses (Challenge ELOQUENT @ CLEF 2026)
**Auteurs** : Paul-Alexandre BAUDRAND, Yanis DEGHEB, Romain BROCHET, Ibrahime CAMARA
**Université Toulouse III — Paul Sabatier — MIAGE M2 Big Data Analytics**

---

## Table des matières

1. [Objectif du Lot C](#1-objectif-du-lot-c)
2. [Rappel du protocole ELOQUENT](#2-rappel-du-protocole-eloquent)
3. [Architecture retenue](#3-architecture-retenue)
4. [Variantes implémentées](#4-variantes-implémentées)
   - 4.1 [Variante 1 — System Prompt](#41-variante-1--system-prompt)
   - 4.2 [Variante 2 — Prefix + Suffix par langue](#42-variante-2--prefix--suffix-par-langue)
   - 4.3 [Variante 3 — Rewrite (reformulation automatique)](#43-variante-3--rewrite-reformulation-automatique)
5. [Traçabilité — `prompt_trace` dans le JSONL de sortie](#5-traçabilité--prompt_trace-dans-le-jsonl-de-sortie)
6. [Configuration et lancement des runs](#6-configuration-et-lancement-des-runs)
7. [Plan d'expérimentation comparable](#7-plan-dexpérimentation-comparable)
8. [Conformité au protocole et au format de soumission ELOQUENT](#8-conformité-au-protocole-et-au-format-de-soumission-eloquent)
9. [Pré-requis pour l'analyse Lot D](#9-pré-requis-pour-lanalyse-lot-d)
10. [Résumé exécutif (extrait pour le rapport final)](#10-résumé-exécutif-extrait-pour-le-rapport-final)
11. [Annexes — Fichiers livrés](#11-annexes--fichiers-livrés)

---

## 1. Objectif du Lot C

L'énoncé impose, **en plus de la baseline obligatoire (Lot A)**, au moins une variante de prompting permettant de tester la **stabilité et la sensibilité aux formulations**. Le sujet recommande explicitement trois familles d'approches :

> - Une variante reposant sur un *system prompt* (consigne globale)
> - Une variante reposant sur un préfixe / suffixe ou une contrainte de style (réponse en une phrase, neutralité, etc.)
> - Une stratégie de reformulation automatique des prompts avant envoi au modèle

Le Lot C livre **les trois** :

| Variante | Mécanisme | Provider cible | Hypothèse testée |
|---|---|---|---|
| C1 — `system_prompt` | Consigne unique en `role: system` | Groq llama-3.1-8b-instant | Une consigne globale réduit la dispersion sur l'axe `specific` (robustesse). |
| C2 — `prefix_suffix` | Texte ajouté avant/après la question, traduit par langue | Qwen 2.5 3B local (Ollama) | Sur petits modèles ouverts qui suivent moins bien les `system`, un wrapping textuel par langue améliore la cohérence linguistique. |
| C3 — `rewrite` | Question paraphrasée par un LLM tiers avant envoi au modèle cible | Qwen 2.5 3B local + Groq comme rewriter | Faire reformuler par un modèle plus puissant lève les ambiguïtés implicites et explicite le contexte culturel sous-entendu. |

**Contrainte transverse imposée par le sujet :** chaque variante doit être **traçable** — on doit pouvoir savoir exactement quelle transformation a été appliquée et avec quels paramètres. C'est l'objet du dispositif `prompt_trace` décrit en §5.

---

## 2. Rappel du protocole ELOQUENT

Pour bien situer le Lot C, voici les contraintes du challenge qui s'appliquent à **toutes** les variantes :

- **Une question = une session indépendante** : pas d'historique entre les questions.
- **Réponses courtes** (~ une phrase).
- **Run déterministe** obligatoire pour la baseline (`temperature = 0`). Les variantes du Lot C **conservent** ce déterminisme — la transformation porte sur le prompt, jamais sur les paramètres de génération.
- **Deux types de datasets** :
  - `unspecific` (Cultural **Diversity**) : la culture est inférée à partir de la langue.
  - `specific` (Cultural **Robustness**) : le pays / contexte est explicitement indiqué.
- **Format d'entrée** : un fichier `<lang>_<type>.jsonl` par couple langue × type.
- **Format de sortie** : le même JSONL avec un champ `answer` ajouté.
- **Métadonnées de soumission** : un fichier annexe au format documenté par le challenge (cf. §8).

Le Lot C ne change rien à ces invariants : il agit **uniquement** entre la lecture de la question et l'appel au LLM.

---

## 3. Architecture retenue

Toutes les stratégies partagent le même contrat (`Protocol` Python). Le pipeline du Lot A est inchangé dans son flux global ; seul le constructeur de messages devient pluggable.

```
                          ┌────────────────────────────────────┐
JSONL d'entrée ──┐        │  PromptStrategy.build(question, lang) │
(une question)   │        │   ↓                                 │
                 ▼        │   ┌─────────┐  ┌───────────────┐    │
        PipelineRunner ──►│   │messages │  │prompt_trace   │    │
                 ▲        │   └────┬────┘  └───────┬───────┘    │
                 │        │        │               │            │
                 │        └────────┼───────────────┼────────────┘
                 │                 ▼               │
                 │          provider.generate()    │
                 │                 │               │
                 ▼                 ▼               ▼
            Record JSONL ◄── { ..., "answer": ..., "prompt_trace": {...} }
```

**Décisions de conception clés :**

1. **Un Protocol unique `PromptStrategy`** avec une méthode `build(question, lang) → PromptBuildResult`. La langue est passée explicitement parce qu'elle est requise pour `prefix_suffix` ; les autres stratégies l'ignorent.
2. **`PromptBuildResult`** porte trois champs :
   - `messages` : prêts pour `provider.generate()` ;
   - `trace` : dict décrivant la transformation appliquée ;
   - `rewritten_text` : optionnel, utilisé par la variante `rewrite`.
3. **Le rewriter de la variante C3 est un `LLMProvider`** — exactement la même abstraction que les modèles cibles. On peut donc combiner librement : Groq comme rewriter + Qwen comme cible, ou Qwen comme rewriter + Groq comme cible, etc.
4. **Pas de globals, pas d'état caché** : tout passe par la config YAML, validée au chargement. Une config = un run reproductible.
5. **Modifications chirurgicales** : `pipeline.py` n'a été touché qu'aux deux endroits qui le nécessitaient strictement (constructeur de stratégie + ajout de `prompt_trace` dans le record).

---

## 4. Variantes implémentées

### 4.1 Variante 1 — System Prompt

**Fichier de config :** `configs/variant_c1_system_prompt.yaml`

**Principe :** Une consigne globale est placée en `role: "system"` avant chaque question. L'instruction reste **strictement orthogonale au contenu de la question** — on ne souffle aucun élément culturel ni géographique. On contraint uniquement :
- la langue de réponse (alignée sur la question)
- la longueur (une phrase courte)
- la neutralité ou la sensibilité culturelle, selon le preset.

**Trois presets prêts à l'emploi**, exposés via la clé `preset:` du YAML :

| Preset | Intention |
|---|---|
| `concise` | Concision stricte (≤ 25 mots), pas de disclaimers, pas de méta-commentaires. |
| `neutral` | Réponse factuelle, évite stéréotypes et opinions personnelles. |
| `culturally_aware` | Si la question implique un contexte culturel, ancrer la réponse dans **ce** contexte plutôt que de donner une réponse générique mondiale. |

L'utilisateur peut aussi fournir son propre system prompt via la clé `system_prompt:` (qui prime sur `preset:` si les deux sont fournis).

**Exemple de transformation :**

```
Question d'entrée  : "Quel est le plat traditionnel pour Noël ?"
                     (fr_specific.jsonl, contexte = France)

Messages envoyés au LLM :
  [
    {"role": "system",  "content": "<preset 'culturally_aware'>"},
    {"role": "user",    "content": "Quel est le plat traditionnel pour Noël ?"}
  ]

Hypothèse Lot D : sur l'axe `specific`, le modèle ancre davantage la
réponse dans le contexte (dinde aux marrons / bûche de Noël) plutôt que
de fournir un panorama mondial générique.
```

---

### 4.2 Variante 2 — Prefix + Suffix par langue

**Fichier de config :** `configs/variant_c2_prefix_suffix.yaml`

**Principe :** Plutôt que de poser la consigne en `role: system`, on l'**injecte directement dans le texte utilisateur** sous forme de préfixe et de suffixe — **traduits dans la langue de la question**.

**Pourquoi cette duplication apparente avec C1 ?** Les petits modèles ouverts (Qwen 2.5 3B, Llama 3B, Phi-3) suivent typiquement moins bien les `system` que les modèles plus gros. Un wrapping textuel **dans la même langue que la question** s'avère plus robuste pour ces modèles. C1 et C2 testent donc **le même objectif fonctionnel par deux mécanismes différents** — c'est exactement la « comparaison structurée » demandée par le sujet.

**Préfixes / suffixes par défaut** (couvrant les 5 langues du Lot A) :

| Lang | Préfixe (default) | Suffixe (default) |
|---|---|---|
| fr | `Réponds en une seule phrase courte, en français : ` | `\n\nRéponse (une phrase) :` |
| en | `Answer in one short sentence, in English: ` | `\n\nAnswer (one sentence):` |
| es | `Responde en una sola frase corta, en español: ` | `\n\nRespuesta (una frase):` |
| it | `Rispondi in una sola frase breve, in italiano: ` | `\n\nRisposta (una frase):` |
| de | `Antworte in einem einzigen kurzen Satz auf Deutsch: ` | `\n\nAntwort (ein Satz):` |

**Override partiel possible** : l'utilisateur peut surcharger une seule langue dans le YAML — les autres conservent les défauts (merge dict). Une langue absente de la table → texte intact (transparent).

**Exemple de transformation :**

```
Question d'entrée : "¿Cuál es la capital de España?"

Message envoyé :
  [
    {"role": "user", "content":
       "Responde en una sola frase corta, en español: ¿Cuál es la capital de España?\n\nRespuesta (una frase):"}
  ]
```

---

### 4.3 Variante 3 — Rewrite (reformulation automatique)

**Fichier de config :** `configs/variant_c3_rewrite.yaml`

**Principe :** Chaque question passe d'abord dans un **rewriter LLM** (qui peut être un modèle plus puissant que la cible), avec une instruction claire : *normaliser, lever les ambiguïtés, expliciter le contexte culturel s'il était implicite, garder une seule phrase*. La version réécrite est **ensuite** envoyée au modèle cible — **dans la même langue**.

**Architecture en deux étapes** :

```
question_originale ──► rewriter LLM (Groq llama-3.1-8b)
                       │  prompt = "Rewrite this question in {lang}.
                       │             Goals: remove ambiguity, make
                       │             cultural context explicit, ..."
                       ▼
                  question_réécrite
                       │
                       ▼
                  modèle cible (Qwen 2.5 3B local)
                       │
                       ▼
                       answer
```

**Configuration recommandée :**

```yaml
prompting:
  strategy: "rewrite"
  rewriter:
    provider: "groq"                # rewriter puissant + rapide
    model: "llama-3.1-8b-instant"
    max_tokens: 80
```

**Garde-fous d'ingénierie :**

1. **Fallback sur la question originale** si le rewriter échoue (timeout, quota, réponse vide). C'est la `trace.rewriter_status = "fallback_original"` qui le signale dans le JSONL — pas de question manquante dans le run.
2. **Déterminisme préservé** : le rewriter tourne aussi à `temperature = 0`.
3. **Sauvegarde de l'original ET de la réécriture** dans `prompt_trace`. C'est essentiel pour le Lot D : on doit pouvoir mesurer le delta sémantique introduit par le rewriter, pas seulement constater l'effet final.
4. **Coût documenté** : 2 appels LLM par question. Sur le dataset complet (5 langues × ~500 questions), prévoir ~2× le temps d'un run baseline.

**Exemple concret :**

```
Question d'entrée  : "C'est quoi le repas typique ?"
                     (fr_specific.jsonl, contexte = France implicite)

Sortie rewriter    : "Quel est le repas typique en France pour le déjeuner ?"

Messages envoyés au modèle cible :
  [
    {"role": "user", "content":
       "Quel est le repas typique en France pour le déjeuner ?"}
  ]
```

---

## 5. Traçabilité — `prompt_trace` dans le JSONL de sortie

Chaque ligne du JSONL produit par le pipeline contient désormais un nouveau champ `prompt_trace` qui décrit **exactement** ce qui a été fait. C'est le mécanisme central de traçabilité exigé par le sujet :

> *« Chaque variante doit être traçable : on doit pouvoir savoir exactement quelle transformation a été appliquée et avec quels paramètres. »*

### Structure de `prompt_trace` selon la stratégie

| Stratégie | Clés présentes |
|---|---|
| `vanilla` | `strategy` |
| `system_prompt` | `strategy`, `preset`, `system_prompt` |
| `prefix_suffix` | `strategy`, `lang`, `prefix`, `suffix` |
| `rewrite` | `strategy`, `lang`, `rewriter_provider`, `rewriter_status`, `rewriter_latency_ms`, `original_text`, `rewritten_text` |

### Exemple — record produit par la variante C1

```json
{
  "id": "fr-spec-001",
  "country": "FR",
  "question": "Quel est le plat traditionnel pour Noël ?",
  "answer": "En France, le plat traditionnel de Noël est la dinde aux marrons accompagnée d'une bûche en dessert.",
  "prompt_trace": {
    "strategy": "system_prompt",
    "preset": "culturally_aware",
    "system_prompt": "You are a culturally aware assistant. Always answer in the same language..."
  }
}
```

### Exemple — record produit par la variante C3

```json
{
  "id": "fr-spec-042",
  "question": "C'est quoi le repas typique ?",
  "answer": "Le repas typique français pour le déjeuner est composé d'une entrée, d'un plat principal et d'un dessert.",
  "prompt_trace": {
    "strategy": "rewrite",
    "lang": "fr",
    "rewriter_provider": "groq",
    "rewriter_status": "ok",
    "rewriter_latency_ms": 312.4,
    "original_text": "C'est quoi le repas typique ?",
    "rewritten_text": "Quel est le repas typique en France pour le déjeuner ?"
  }
}
```

### Conservation au-delà du JSONL

En plus de `prompt_trace`, **chaque dossier de run** contient déjà :

- `config_snapshot.yaml` : la config exacte utilisée (sans la clé API) ;
- `run_metadata.json` : provider, modèle, dates, durées, stats par langue.

Donc à partir d'un seul fichier JSONL produit par le pipeline, on peut **reproduire à l'identique** la transformation : la stratégie + ses paramètres sont écrits dans la trace, et la config complète est dans le dossier de run.

---

## 6. Configuration et lancement des runs

### Pré-requis (déjà en place dans le Lot A)

```bash
# Activation de l'environnement
.venv\Scripts\activate          # Windows

# Dépendances (rien à ajouter pour le Lot C — uniquement les dépendances du Lot A)
pip install -e ".[dev]"

# Pour la variante C2 (Qwen local) et la variante C3 (cible Qwen)
ollama pull qwen2.5:3b

# Pour la variante C1 et le rewriter de la C3 (Groq)
# .env doit contenir GROQ_API_KEY=gsk_...
```

### Lancement

```bash
# Variante 1 — System prompt
python run.py --config configs/variant_c1_system_prompt.yaml

# Variante 2 — Prefix + suffix
python run.py --config configs/variant_c2_prefix_suffix.yaml

# Variante 3 — Rewrite (Groq comme rewriter, Qwen comme cible)
python run.py --config configs/variant_c3_rewrite.yaml
```

### Sortie produite

Pour chaque run, le pipeline crée un dossier daté dans `data/output/runs/` :

```
data/output/runs/variant_c1_groq_system_prompt_20260428_143022/
├── config_snapshot.yaml          ← config exacte (sans clé API)
├── run_metadata.json             ← stats, durées, erreurs par langue
├── fr_specific_output.jsonl      ← une ligne = une question + answer + prompt_trace
├── it_specific_output.jsonl
├── en_specific_output.jsonl
├── es_specific_output.jsonl
└── de_specific_output.jsonl
```

### Astuce — limiter la durée pendant le développement

Conformément à la décision déjà prise dans le Lot A (cf. compte-rendu de réunion : *« ne pas exécuter de run, prend beaucoup trop de temps pour le moment »*), on peut :
- soit limiter le dataset (réduire les langues dans le YAML, garder seulement `fr`) ;
- soit faire tourner localement via Qwen plutôt que Groq (pas de quota gratuit) ;
- soit pointer `paths.input_dir` vers un sous-dataset des 50 premières questions.

---

## 7. Plan d'expérimentation comparable

Pour que le Lot D puisse comparer baseline et variantes **dans des conditions strictement comparables**, on garantit :

| Paramètre | Baseline (Lot A) | C1 system_prompt | C2 prefix_suffix | C3 rewrite |
|---|---|---|---|---|
| Modèle cible | Groq llama-3.1-8b-instant | **Idem** | Qwen 2.5 3B | Qwen 2.5 3B |
| `temperature` | 0.0 | 0.0 | 0.0 | 0.0 |
| `max_tokens` | 150 | 150 | 150 | 150 |
| `top_p` | 1.0 | 1.0 | 1.0 | 1.0 |
| Langues | fr, it, en, es, de | **idem** | **idem** | **idem** |
| Datasets | `specific` + `unspecific` (à lancer 2× chaque variante) | **idem** | **idem** | **idem** |
| Sessions | indépendantes | **idem** | **idem** | **idem** |

**Pour le rapport final**, on couvre **deux paires de comparaisons légitimes** :

1. **Baseline Groq vs C1 (Groq + system prompt)**
   → Mesure l'effet pur du *system prompt* sur un même modèle. C'est l'« A/B test » classique demandé par le sujet.

2. **Baseline Qwen vs C2 (Qwen + prefix/suffix) vs C3 (Qwen + rewrite via Groq)**
   → Mesure deux mécanismes distincts d'amélioration **du même modèle de base** (Qwen 3B). C2 = transformation locale ; C3 = transformation par un modèle plus puissant.

Ces deux axes répondent aux deux questions du sujet : **stabilité aux formulations** (C1 vs baseline) et **sensibilité aux mécanismes de prompting** (C2 vs C3 vs baseline Qwen).

---

## 8. Conformité au protocole et au format de soumission ELOQUENT

Le sujet impose (et la page ELOQUENT confirme) un **format de métadonnées de soumission**. Le snapshot ci-dessous (extrait du PDF du projet, page 4) cible précisément les variantes :

```json
{
  "team": "your-team-name",
  "system": "your-system-name",
  "model": "model-identifier",
  "submissionid": "experiment-1",
  "date": "2026-05-15",
  "label": "eloquent-2026-cultural",
  "languages": ["en", "de", "fr", "sv", "ru"],
  "modifications": {
    "system_prompt":         "You are a culturally aware assistant...",
    "prompt_prefix_english": "Context: ...",
    "prompt_suffix_english": " Please be specific.",
    "generation_params":     {"do_sample": false, "max_new_tokens": 200},
    "notes":                 "Testing impact of cultural awareness system prompt version 1"
  }
}
```

### Comment nos variantes alimentent ce JSON de soumission

| Champ ELOQUENT | Source dans nos runs |
|---|---|
| `team`, `system`, `submissionid`, `date`, `label` | À renseigner par le Lot B au moment de l'export. |
| `model` | `run_metadata.json` → `model` |
| `languages` | `run_metadata.json` → `languages` |
| `modifications.system_prompt` | C1 : `prompt_trace.system_prompt` (identique pour tout le run). |
| `modifications.prompt_prefix_<lang>` / `suffix_<lang>` | C2 : `prompt_trace.prefix` / `prompt_trace.suffix` regroupés par langue. |
| `modifications.generation_params` | `run_metadata.json` → `generation` (`temperature`, `max_tokens`, `top_p`). |
| `modifications.notes` | Phrase courte décrivant l'hypothèse — à éditer manuellement avant soumission. |

**Pour la variante C3 (rewrite)** — qui n'est pas littéralement couverte par les champs `prompt_prefix_*` du template — on encode la stratégie sous le champ `notes` et on ajoute un sous-objet `rewriter` (provider + modèle) dans `modifications`. Ça reste lisible par les organisateurs et ça documente fidèlement ce qui a été fait, ce qui est l'esprit du champ.

---

## 9. Pré-requis pour l'analyse Lot D

Pour que l'analyse quantitative et qualitative du Lot D soit possible, le Lot C garantit :

1. **Comparabilité 1-pour-1** : chaque record d'une variante a le même `id` que le record correspondant de la baseline. Ça permet de joindre baseline et variantes par `id` × `lang` et de comparer phrase à phrase.
2. **Métadonnées préservées** : tous les champs originaux du JSONL d'entrée (typiquement `id`, `country`, `question`, parfois `topic`) sont **conservés** dans la sortie — on ne fait qu'ajouter `answer` et `prompt_trace`.
3. **Texte original ET texte transformé sauvegardés** pour la C3, ce qui permet à l'analyse Lot D de mesurer :
   - le delta sémantique introduit par le rewriter (embedding original vs réécriture) ;
   - le delta de réponse induit chez le modèle cible (réponse baseline vs réponse via rewriter).
4. **Stats brutes du run** dans `run_metadata.json` : `total`, `success`, `errors`, `avg_latency_ms` par langue. C'est suffisant pour les statistiques simples demandées (longueur, taux de réponses vides, erreurs) sans aucun parse supplémentaire.
5. **Cohérence linguistique mesurable** : le Lot D pourra, par langue, vérifier si la réponse est bien dans la langue attendue (cf. la note du PDF : *« Évaluer si quand on fait une demande en français il répond en français »*). Cette vérification ne dépend pas de la stratégie : elle se fait sur le champ `answer` final.

---

## 10. Résumé exécutif (extrait pour le rapport final)

> Le Lot C ajoute trois variantes de prompting sur la chaîne d'évaluation ELOQUENT, sans modifier le pipeline backend du Lot A au-delà des deux points d'extension nécessaires :
>
> 1. **`system_prompt`** — consigne globale en `role: system`, en trois presets (concise, neutral, culturally_aware) ou personnalisée. Testée sur Groq llama-3.1-8b-instant.
> 2. **`prefix_suffix`** — wrapping textuel par langue, ajouté autour de la question. Testée sur Qwen 2.5 3B local. Ciblée pour les petits modèles ouverts qui suivent moins bien les `system` prompts.
> 3. **`rewrite`** — reformulation automatique de chaque question par un LLM tiers (typiquement plus puissant) avant envoi au modèle cible. Avec fallback sur le texte original en cas d'échec du rewriter.
>
> Chaque variante est entièrement déclarative (un fichier YAML), strictement déterministe (`temperature = 0`) et entièrement traçable : un nouveau champ `prompt_trace` est ajouté à chaque record du JSONL de sortie et décrit exactement la transformation appliquée — y compris, pour `rewrite`, le texte original et le texte réécrit côte à côte. Combiné au snapshot de config et aux métadonnées du run, ce champ rend chaque expérience reproductible à partir d'un unique dossier de sortie.
>
> Le périmètre couvre les 5 langues retenues par le projet (fr, it, en, es, de) sur les deux types de datasets ELOQUENT (`specific` pour la robustesse culturelle, `unspecific` pour la diversité). Le plan d'expérimentation prévoit deux axes de comparaison : Groq baseline vs Groq + system_prompt (effet pur d'un system prompt), et Qwen baseline vs Qwen + prefix_suffix vs Qwen + rewrite (deux mécanismes distincts pour améliorer un même petit modèle).

---

## 11. Annexes — Fichiers livrés

### Code modifié

| Fichier | Nature de la modification |
|---|---|
| `src/eloquent/prompting.py` | Réécrit pour porter `PromptBuildResult`, les 4 stratégies et la factory. |
| `src/eloquent/config.py` | Étendu avec `RewriterConfig` et les paramètres optionnels de chaque variante dans `PromptingParams` ; validation enrichie ; `to_dict()` sérialise les nouveaux champs (sans clé API). |
| `src/eloquent/pipeline.py` | Deux modifications minimales : `_build_strategy()` instancie la bonne stratégie depuis la config (avec création du rewriter pour C3), et `_process_record()` ajoute `prompt_trace` au record de sortie. |

### Fichiers ajoutés

| Fichier | Rôle |
|---|---|
| `configs/variant_c1_system_prompt.yaml` | Config de la variante 1 — Groq + system prompt (preset `concise`). |
| `configs/variant_c2_prefix_suffix.yaml` | Config de la variante 2 — Qwen local + prefix/suffix par langue. |
| `configs/variant_c3_rewrite.yaml` | Config de la variante 3 — Qwen local + rewriter Groq. |
| `docs/LOT_C_RECAP.md` | Le présent document. |

### Fichiers du Lot A inchangés

`src/eloquent/providers.py`, `src/eloquent/logger.py`, `run.py`, `tests/`, `pyproject.toml`, `configs/baseline_groq.yaml`, `configs/baseline_qwen.yaml`. Ces fichiers ne sont pas touchés par le Lot C : les deux nouveaux points d'extension (`build_strategy()` paramétré et `prompt_trace` dans le record) suffisent à porter les trois variantes.
