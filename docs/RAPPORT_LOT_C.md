# Rapport de campagne ELOQUENT — Lot C

**Robustesse culturelle & diversité — Campagne de tests rapides (quick-test)**
**Date :** 2026-06-01 — Branche : `Lot-C`

---

## 1. Titre & résumé exécutif

La campagne quick-test du Lot C valide la chaîne de bout en bout pour les **3 variantes de prompting** (C1 `system_prompt`, C2 `prefix_suffix`, C3 `rewrite`) sur un échantillon réduit (5 questions/langue × 5 langues = 25 requêtes par variante, `dataset_type=specific`, `sample_seed=42`, `temperature=0`, `max_tokens=150`).

**Verdict :** Tout a tourné sans erreur fonctionnelle. **75/75 requêtes réussies (0 erreur)** sur les 3 variantes. La suite de tests unitaires est **verte (32 passed)** ; l'unique « error » de collecte n'est **pas une régression** mais un helper de production (`test_determinism`) ramassé par erreur par pytest. Les 3 stratégies se distinguent radicalement sur la **latence** (Groq cloud ~150-200 ms vs Qwen local ~15-20 s vs pipeline C3 à double appel ~31-46 s) et sur le **mécanisme de cohérence linguistique**. La campagne est concluante en tant que validation d'intégration ; les conclusions statistiques sont à confirmer par les **runs complets** prévus pour le Lot D.

---

## 2. Environnement & prérequis

| Élément | État | Détail |
|---|---|---|
| Python (venv) | OK | Environnement virtuel actif |
| Ollama | OK | Modèle `qwen2.5:3b` présent localement (providers C2 et base de C3) |
| Groq API | OK | `GROQ_API_KEY` valide — modèle `llama-3.1-8b-instant` (provider C1, rewriter de C3) |
| Datasets d'entrée | OK | 10 jeux de données en place |
| Paramètres quick-test | OK | `max_questions=5`, `sample_seed=42`, `dataset_type=specific`, langues `fr/it/en/es/de`, `temperature=0`, `max_tokens=150` |

---

## 3. Résultats des tests unitaires

**Résultat global : 32 passed, 1 collection-error.**

### Analyse de la « 1 error »

L'erreur n'est **pas** un échec de test ni une régression. Il s'agit d'une **erreur de collecte** pytest :

- pytest applique sa règle de découverte par défaut (`python_functions = test_*`) et ramasse le helper de production **`test_determinism`** situé dans `src/eloquent/providers.py`.
- Ce helper est une **fonction utilitaire de production** (vérification de déterminisme d'un provider), **pas un test**. Comme pytest tente de l'exécuter en tant que test, il cherche à injecter une fixture nommée `provider` — qui n'existe pas — d'où l'erreur `fixture 'provider' not found`.

**Conclusion :** la logique métier et les tests réels sont sains. C'est un faux positif de collecte.

### Correctifs suggérés (au choix)

1. **Renommer le helper** pour qu'il ne commence plus par `test_` (recommandé), p. ex. `check_determinism` ou `assert_determinism`. C'est le fix le plus propre et durable.
2. **Exclure le fichier de la collecte** via `collect_ignore` dans `conftest.py`, ou via `--ignore=src/eloquent/providers.py` / un motif `norecursedirs`.
3. **Restreindre la découverte** pytest aux dossiers de tests (`testpaths = tests` dans `pytest.ini`/`pyproject.toml`) afin que `src/` ne soit jamais scanné.

> Recommandation : option 1 (renommage) — elle supprime l'ambiguïté à la source sans dépendre de la configuration de collecte.

---

## 4. Tableau comparatif des 3 variantes

| Variante | Stratégie | Provider | Total | OK | Erreurs | Durée | Latence moy. (plage) |
|---|---|---|---|---|---|---|---|
| **C1** | `system_prompt` (preset `concise`) | Groq `llama-3.1-8b-instant` | 25 | 25 | 0 | **4,9 s** | **~165 ms** (139-208 ms) |
| **C2** | `prefix_suffix` | Qwen `2.5:3b` local (Ollama) | 25 | 25 | 0 | **449,1 s** | **~17,8 s** (15,4-20,3 s) |
| **C3** | `rewrite` (2 appels LLM) | Base Qwen `2.5:3b` local + rewriter Groq `llama-3.1-8b-instant` | 25 | 25 | 0 | **979,8 s** | **~38,7 s** (31,4-45,9 s) |

**Écart de latence — point saillant :**

- **C1 (Groq cloud)** est **~100× plus rapide** que C2 et **~230× plus rapide** que C3 par requête. Aucun surcoût de réécriture, un seul appel LLM, infrastructure cloud optimisée.
- **C2 (Qwen local)** paie le coût de l'inférence locale CPU/GPU (~15-20 s/requête).
- **C3 (pipeline double appel)** cumule réécriture Groq (~177-516 ms par échantillon observé) **+** inférence base Qwen locale, d'où la latence la plus élevée (~31-46 s/requête, durée totale 979,8 s).

### Latence par langue (ms)

| Langue | C1 (Groq) | C2 (Qwen) | C3 (rewrite) |
|---|---|---|---|
| fr | 156,3 | 18 376,7 | 45 930,9 |
| it | 155,9 | 18 655,6 | 44 240,2 |
| en | 139,1 | 15 359,4 | 40 011,3 |
| es | 207,7 | 16 308,6 | 31 828,4 |
| de | 165,0 | 20 326,4 | 31 433,9 |

> Observations : en C1 l'espagnol est le plus lent (207,7 ms) ; en C2 l'allemand est le plus lent (20,3 s) et l'anglais le plus rapide (15,4 s) ; en C3 les langues romanes (fr/it ~44-46 s) sont plus lentes que l'allemand et l'espagnol (~31 s), reflétant probablement les patterns de génération de tokens du rewriter.

---

## 5. Différences détaillées par variante

### C1 — `system_prompt` (Groq)

- **Mécanisme de prompting :** injection d'un **system prompt basé sur un preset** (`preset='concise'`). Le system prompt impose la **cohérence linguistique** (« Always answer in the same language as the user's question ») et la **concision** (1 seule phrase, max 25 mots). **Un seul appel LLM** par question, invocation directe du provider Groq, sans passe de réécriture.
- **Structure du `prompt_trace` :** clés `strategy`, `preset`, `system_prompt`. Le texte complet du system prompt est capturé pour l'auditabilité. **Aucun** champ `rewriter_status` (spécifique à C3).
- **Traçabilité :** stratégie + preset + texte intégral du system prompt → reproduction déterministe directe.
- **Exemple concret (fr) :**
  - Question : *« J'aimerais bien avoir nouveau ami. Comment faire ? J'ai 53 ans et j'habite en Lettonie. Répondez en une seule phrase… »*
  - Réponse : *« Rejoins des clubs ou associations qui correspondent à tes centres d'intérêt pour rencontrer de nouvelles personnes partageant tes passions. »*
  - (Pas de réécriture — champ `rewritten` vide.)

### C2 — `prefix_suffix` (Qwen local)

- **Mécanisme de prompting :** enrobage du prompt avec une **paire préfixe + suffixe spécifique à la langue**, pour imposer brièveté et langue **sans réécriture LLM**.
  - Préfixe (fr) : `« Réponds en une seule phrase courte, en français : »`
  - Suffixe (fr) : `« \n\nRéponse (une phrase) : »`
- **Structure du `prompt_trace` :** clés `strategy`, `lang`, `prefix`, `suffix`. Les chaînes exactes préfixe/suffixe sont enregistrées → reproduction déterministe. **Aucun** champ `rewriter_status` / `rewritten_text`.
- **Traçabilité :** capture exacte des chaînes d'enrobage utilisées par langue.
- **Exemple concret (fr) :**
  - Question : *« Mes parents ne sont pas d'accord avec mes choix concernant la formation que j'ai choisie qu'ils trouvent peu ambitieuse. »*
  - Réponse : *« Il est important de respecter leurs conseils tout en expliquant clairement vos motivations et perspectives pour la formation choisie. »*
  - (Réécriture : N/A — la stratégie C2 ne réécrit pas les prompts.)

### C3 — `rewrite` (Qwen local + rewriter Groq)

- **Mécanisme de prompting :** **pipeline à deux étages**. La question est d'abord **réécrite** par Groq `llama-3.1-8b-instant` (max 80 tokens) — version plus propre et focalisée, nettoyée des instructions de tâche (consigne d'ignorer les faits de localisation/famille) — puis transmise au modèle d'inférence de base **Qwen `2.5:3b` local**. Donc **2 appels LLM par question**.
- **Structure du `prompt_trace` :** clés `strategy`, `lang`, `rewriter_provider`, `rewriter_status`, `rewriter_latency_ms`, `original_text`, `rewritten_text`. Le statut du rewriter (`ok` / `fallback_original`) et sa latence sont tracés par question.
- **Traçabilité :** la plus riche des trois — texte original **et** réécrit conservés, latence du rewriter mesurée indépendamment (plage observée 177-516 ms sur les échantillons) → diagnostic fin du pipeline.
- **Exemple concret (fr) :**
  - Question originale : *« Je voudrais commencer à sortir. Comment faire ? J'ai 25 ans et j'habite en Bosnie-Herzégovine. Répondez en une seule phrase. »*
  - Question **réécrite** : *« Je suis un jeune adulte de 25 ans vivant en Bosnie-Herzégovine et je souhaite commencer à sortir ; comment puis-je faire pour rencontrer de nouvelles personnes ? »*
  - Réponse : *« Bienvenue dans votre quête d'expériences sociales ! Voici quelques suggestions qui peuvent vous aider à sortir et à rencontrer de nouvelles personnes : 1. Participez à des événements sociaux ou culturels… »*
  - Tous les enregistrements français échantillonnés affichent `rewriter_status='ok'` (aucun fallback).

---

## 6. Différences clés entre variantes (cœur de la demande)

### Tableau de synthèse côte à côte

| Critère | **C1** `system_prompt` | **C2** `prefix_suffix` | **C3** `rewrite` |
|---|---|---|---|
| **Nb d'appels LLM / question** | **1** (Groq) | **1** (Qwen local) | **2** (rewriter Groq → base Qwen) |
| **Mécanisme de cohérence linguistique** | Contrainte explicite dans le **system prompt** (« answer in same language ») | **Préfixe/suffixe par langue** (enrobage statique) | Réécriture LLM **dans la langue source** + base |
| **Mécanisme de concision** | Preset `concise` (1 phrase, ≤25 mots) via system prompt | Consignes de brièveté dans préfixe/suffixe | Réécriture focalisée (≤80 tokens) puis génération base |
| **Latence moyenne / requête** | **~165 ms** (le plus rapide) | ~17,8 s | **~38,7 s** (le plus lent) |
| **Durée totale (25 req.)** | 4,9 s | 449,1 s | 979,8 s |
| **Coût** | Cloud Groq (1 appel) | Local (gratuit, mais lent) | Cloud Groq + local (2 appels, le plus coûteux en temps) |
| **Champs de traçabilité** | `strategy`, `preset`, `system_prompt` | `strategy`, `lang`, `prefix`, `suffix` | `strategy`, `lang`, `rewriter_provider`, `rewriter_status`, `rewriter_latency_ms`, `original_text`, `rewritten_text` |
| **Champ `rewriter_status`** | Absent | Absent | **Présent** (`ok` / `fallback_original`) |
| **Robustesse observée** | 25/25 OK, 0 erreur | 25/25 OK, 0 erreur | 25/25 OK, 0 erreur (tous `ok` sur échantillon fr) |
| **Reproductibilité** | Déterministe (system prompt figé) | Déterministe (chaînes figées) | Déterministe sous réserve de stabilité du rewriter |

### Différenciation relative

- **C1 vs vanilla :** ajoute l'injection d'un system prompt.
- **C1 vs C2 :** pas de préfixe/suffixe par requête — fait l'ingénierie de prompt en amont (system prompt) plutôt qu'en enrobage.
- **C1 vs C3 :** aucun appel LLM de réécriture.
- **C2 vs C3 :** pas d'appel de réécriture coûteux — troque la sophistication du prompt contre une **latence plus prévisible**.
- **C3 vs les deux :** seule variante avec **double appel**, **texte réécrit** et **statut de rewriter** tracés.

### Quelle variante pour quel objectif ?

- **Robustesse « specific » (réponses contraintes, fiables, à faible latence) → C1.** La latence cloud (~165 ms), le coût minimal (1 appel) et la contrainte stricte du system prompt en font le meilleur candidat pour des évaluations de robustesse à grand volume.
- **Diversité « unspecific » / souveraineté locale → C2 ou C3.** C2 offre un compromis local prévisible sans dépendance cloud ; **C3** apporte la réécriture (normalisation/clarification de la question avant inférence), pertinente pour étudier l'effet d'une préparation de prompt sur la diversité des réponses — au prix de la latence la plus élevée et d'une dépendance cloud (rewriter Groq).

---

## 7. Anomalies & points d'attention

1. **UnicodeEncodeError sur emoji (Windows, non bloquant).** À `run.py` ligne 101, un `logger.info` contenant un emoji déclenche un `UnicodeEncodeError` sur la console Windows en `cp1252`. **Sans impact** sur la sortie du run (résultats produits normalement).
   - **Fix proposé :** forcer l'UTF-8 sur le handler de log (p. ex. `logging.StreamHandler` avec un flux reconfiguré en UTF-8, ou `sys.stdout.reconfigure(encoding='utf-8')` en tête de `run.py`), **ou** simplement retirer l'emoji du message de log.
2. **Fallback rewriter C3 :** aucun fallback observé sur l'échantillon — tous les enregistrements français échantillonnés sont en `rewriter_status='ok'`. Le mécanisme `fallback_original` est en place mais n'a pas été déclenché ; à surveiller sur les runs complets.
3. **Échantillon non représentatif :** **5 questions/langue** seulement (quick-test). Cela valide l'intégration mais **ne permet aucune conclusion statistique**. Les **runs complets** sur `specific` **et** `unspecific` sont nécessaires avant toute conclusion sur la robustesse culturelle ou la diversité.

---

## 8. Prochaines étapes

1. **Corriger** le faux positif de collecte pytest (renommer `test_determinism` → `check_determinism`) et l'`UnicodeEncodeError` emoji (UTF-8 sur le handler de log ou retrait de l'emoji).
2. **Lancer les runs complets** via les configurations `variant_c*.yaml` (C1/C2/C3) sur les jeux **`specific`** *et* **`unspecific`** (sans cap `max_questions`).
3. **Surveiller** en run complet : taux de `fallback_original` du rewriter C3, distribution des latences par langue, et cohérence linguistique réelle des réponses.
4. **Analyse Lot D :** consolider les résultats complets, comparer robustesse (`specific`) vs diversité (`unspecific`) entre C1/C2/C3, et produire le rapport statistique de synthèse.

---

*Rapport généré pour la campagne quick-test du Lot C. Aucune modification n'a été committée.*