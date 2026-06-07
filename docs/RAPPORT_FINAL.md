# Rapport final — Application multi-LLM pour la robustesse culturelle et la diversité

**Challenge ELOQUENT @ CLEF 2026 — *Cultural Robustness & Diversity***
**Module :** MIAGE M2 — Méthodes analytiques pour le big data
**Auteurs :** Paul-Alexandre BAUDRAND · Yanis DEGHEB · Romain BROCHET · Ibrahime CAMARA
**Dépôt :** `App-multi-LLM` · **Date :** 2026-06-07

> Format : rapport d'expérimentation court. Il couvre le protocole, les modèles, les variantes,
> les résultats de l'analyse (Lot D), une typologie qualitative d'écarts, les limites et des
> recommandations. **Tous les chiffres cités proviennent des artefacts versionnés sous
> `data/output/analysis/`**, régénérés par exécution complète du notebook
> `notebooks/analyse_lot_d_final.ipynb` (`Restart & Run All`, sans erreur), sur **76 545 réponses**.

---

## 1. Résumé exécutif

Nous avons construit une application multi-LLM reproductible qui exécute des **runs** d'évaluation
sur les jeux de données ELOQUENT (`specific` = robustesse culturelle, `unspecific` = diversité
culturelle), pour **5 langues** (fr, it, de, es, en), avec deux modèles et plusieurs stratégies de
prompting. L'analyse compare ces conditions par des statistiques simples et par une **mesure
sémantique fondée sur des embeddings multilingues**.

Trois résultats principaux ressortent, tous mesurés sur les sous-ensembles **strictement alignés
sur 5 langues** :

1. **Le gros modèle est plus robuste entre langues.** Sur `specific` (contexte/pays explicite, on
   attend de la cohérence), Llama 3.1 8B atteint une cohérence sémantique de **0,730**, contre
   **0,644** pour Qwen 2.5 3B. Le modèle de 8 B donne des réponses plus stables d'une langue à
   l'autre quand le contexte est fixé.
2. **Le petit modèle produit plus de diversité.** Sur `unspecific` (culture inférée par la langue,
   on attend de la variation), Qwen 2.5 3B obtient une diversité de **0,329** (vanilla) à **0,416**
   (prefix_suffix), contre **0,228**–**0,242** pour Llama. Qwen varie davantage selon le prisme
   linguistique.
3. **La variante de prompting agit surtout sur la forme, pas sur le fond.** La stratégie
   `prefix_suffix` (Lot C2) réduit fortement la longueur et le taux de réponses « trop longues »
   (Llama, sous-ensemble aligné : `too_long_pct` 21,5 % → 3,0 % ; Qwen : 10,5 % → 0,2 %) tout en **abaissant légèrement
   la cohérence et en augmentant la diversité** — l'enrobage contraint le style sans homogénéiser
   le contenu entre langues.

La cohérence linguistique (« quand on demande en français, répond-il en français ? ») est
excellente partout sauf pour **Qwen vanilla en italien (7,1 %) et espagnol (5,1 %)** de réponses
dans la mauvaise langue — un défaut de robustesse propre au petit modèle, corrigé par la variante
`prefix_suffix` qui force la langue cible.

---

## 2. Protocole expérimental

### 2.1 Données

Le challenge fournit, pour chaque langue, deux fichiers JSONL :

| Type | Objectif évalué | Attendu |
|---|---|---|
| `specific` | **Robustesse culturelle** — le pays / contexte est **explicite** dans la question | Réponses **cohérentes** entre langues (contexte fixé) |
| `unspecific` | **Diversité culturelle** — la culture est **inférée** de la langue | Réponses **différentes** selon la langue |

Langues cibles : **fr, it, de, es, en**. Les `id` des items `specific` encodent à la fois la
question et le contexte culturel (`id = "question-culture"`, ex. `1-1`), ce qui permet d'**aligner
le même item d'une langue à l'autre** sans heuristique fragile.

### 2.2 Conditions de génération

Conformément au protocole ELOQUENT :

- **Session indépendante** : chaque question est traitée isolément, sans historique.
- **Réponse courte** : `max_tokens = 150` (≈ 1 phrase).
- **Déterminisme** : `temperature = 0,0`, `top_p = 1,0`. La baseline est **déterministe** et
  vérifiée par un test codifié (`check_determinism`, voir §3.3).
- **Échantillonnage reproductible** : quand un sous-ensemble est tiré, la graine est décorrélée par
  langue (`sample_seed ^ hash(lang)`) pour ne pas réinterroger les mêmes indices dans toutes les
  langues.

Toute la configuration est centralisée en YAML (`configs/`) et **persistée avec chaque run**
(`config_snapshot.yaml` + `run_metadata.json` dans le dossier de sortie), garantissant qu'on sait
exactement quels paramètres ont produit quels résultats.

---

## 3. Modèles et architecture

### 3.1 Les deux modèles

| Rôle | Modèle | Accès | Taille |
|---|---|---|---|
| **Modèle via API** | **Llama 3.1 8B (Groq)** | API cloud Groq | 8 milliards de paramètres |
| **Modèle ouvert** | **Qwen 2.5 3B** | Ollama, local | 3 milliards de paramètres |

> **Note technique — un seul modèle Llama, deux identifiants d'API.** Les réponses Llama 3.1 8B
> proviennent toutes de l'**infrastructure Groq**, mais sous deux identifiants : l'espagnol via
> l'appel **Groq direct** (`llama-3.1-8b-instant`), et de/en/fr/it via la **passerelle
> OpenRouter → Groq** (`meta-llama/llama-3.1-8b-instruct`), avec routage forcé vers Groq
> (`provider.order = ["Groq"]`, `allow_fallbacks = false`, cf. `experiments/openrouter_groq/`).
> L'inférence sous-jacente est le **même Llama 3.1 8B** ; la double dénomination n'est qu'un
> artefact d'identifiant. Dans ce rapport, nous traitons donc ces réponses comme **un seul modèle**,
> ce qui permet un alignement **strict sur 5 langues** (cf. §5.2).

### 3.2 Architecture logicielle

- **Abstraction `LLMProvider`** (classe abstraite) + deux implémentations concrètes
  (`GroqProvider`, `QwenOllamaProvider`) instanciées par une factory
  (`build_provider_from_config`). Changer de modèle = changer le YAML, sans toucher au code.
- **Pipeline** (`src/eloquent/pipeline.py`) : lit le JSONL d'entrée, applique la stratégie de
  prompting, interroge le provider (`generate_safe` : pas d'exception, l'erreur est journalisée et
  stockée — conforme à la consigne « pas de reprise automatique, simple journal »), écrit le JSONL
  de sortie en ajoutant `answer` **et** `prompt_trace`.
- **Backend FastAPI** (`server.py`) + **interface Gradio** (`app_gradio.py`), découplés via une API
  REST (`/providers`, `/runs`, `/status`, `/download`).

### 3.3 Reproductibilité du déterminisme

`check_determinism()` (dans `providers.py`, appelé par `run.py`) exécute deux fois la même requête
et vérifie l'égalité des réponses — c'est l'implémentation concrète de l'exigence « deux runs = même
réponse » du sujet.

---

## 4. Variantes de prompting (Lot C)

Trois familles de variantes sont **implémentées et tracées** (chaque ligne de sortie porte un
`prompt_trace` décrivant la transformation exacte appliquée) :

| Variante | Stratégie | Mécanisme | Appels LLM / question |
|---|---|---|---|
| **Baseline** | `vanilla` | Texte brut, aucun prompt engineering | 1 |
| **C1** | `system_prompt` | Consigne globale (preset `concise` : 1 phrase, langue cible) | 1 |
| **C2** | `prefix_suffix` | Enrobage préfixe + suffixe **par langue** (brièveté + langue) | 1 |
| **C3** | `rewrite` | Reformulation de la question par un modèle plus fort, puis génération | 2 |

**Périmètre des runs exploités dans l'analyse (transparence).** Les trois familles ont été validées
fonctionnellement en quick-test (Lot C — voir `docs/RAPPORT_LOT_C.md` : 75/75 requêtes réussies,
0 erreur). Pour l'analyse Lot D ci-dessous, les **runs complets committés** couvrent **`vanilla`** et
**`prefix_suffix` (C2)** sur les deux modèles. C1 et C3 n'ont **pas** de run complet committé : leurs
résultats restent ceux du quick-test (latence et mécanisme documentés au Lot C), et ils sont exclus
des comparaisons chiffrées de robustesse/diversité pour ne pas extrapoler.

---

## 5. Résultats — analyse quantitative (Lot D)

L'analyse repose sur le notebook `analyse_lot_d_final.ipynb`. La mesure sémantique utilise les
embeddings multilingues **`paraphrase-multilingual-MiniLM-L12-v2`** (sentence-transformers) ; la
similarité entre langues est le **cosinus moyen par paires** des réponses alignées d'un même item.

- **Cohérence `specific`** = cosinus moyen entre les réponses des différentes langues pour un même
  item à contexte fixé. **↑ = plus robuste** (les langues convergent).
- **Diversité `unspecific`** = `1 − cosinus moyen`. **↑ = plus de diversité** (les langues divergent).

### 5.1 Statistiques simples (par condition)

Source : `data/output/analysis/run_summary.csv` (chiffres reportés **tels quels**, sans
sous-sélection). Total : **76 545 réponses** chargées (`analysis_summary.json`).

| Modèle | Stratégie | Dataset | n | % vides | mots (moy.) | % trop longues | % mauvaise langue |
|---|---|---|---:|---:|---:|---:|---:|
| Llama 3.1 8B | vanilla | specific | 20 700 | 0,13 | 33,1 | 15,1 | 0,16 |
| Llama 3.1 8B | prefix_suffix | specific | 12 420 | 0,22 | 28,0 | 3,0 | 0,03 |
| Qwen 2.5 3B | vanilla | specific | 20 705 | 0,00 | 26,6 | 10,5 | 2,57 |
| Qwen 2.5 3B | prefix_suffix | specific | 20 700 | 0,00 | 16,4 | 0,2 | 0,43 |
| Llama 3.1 8B | vanilla | unspecific | 505 | 0,00 | 35,6 | 24,2 | 0,00 |
| Llama 3.1 8B | prefix_suffix | unspecific | 505 | 0,00 | 29,0 | 6,7 | 0,00 |
| Qwen 2.5 3B | vanilla | unspecific | 505 | 0,00 | 24,6 | 4,4 | 0,59 |
| Qwen 2.5 3B | prefix_suffix | unspecific | 505 | 0,00 | 15,8 | 0,2 | 0,20 |

> **Note de périmètre (Llama vanilla specific).** Cette ligne agrège les deux identifiants Groq
> (16 560 réponses de/en/fr/it via OpenRouter→Groq + 4 140 en es via Groq direct = 20 700). Les
> valeurs ci-dessus sont les **agrégats de l'artefact** (mots 33,1 ; trop longues 15,1 %). Pour la
> comparaison alignée 5-langues du §5.4, le bloc strictement aligné donne 34,9 mots et 21,5 % de
> réponses trop longues (cf. `baseline_vs_variant.csv`) ; l'écart vient du périmètre d'items, pas
> d'un recalcul.

**Lecture.**
- **Taux de vides quasi nul partout** (≤ 0,22 %) : les runs sont propres et complets.
- **Qwen répond plus court** (≈ 16–27 mots) que Llama (≈ 28–36) ; la consigne « réponse courte » est
  mieux respectée par Qwen, surtout en `prefix_suffix`.
- **La variante `prefix_suffix` discipline la longueur** : le taux de réponses « trop longues »
  s'effondre (Llama 15,1 % → 3,0 % ; Qwen 10,5 % → 0,2 %).

### 5.2 Mesure sémantique — cohérence (specific) et diversité (unspecific)

Source : `data/output/analysis/analysis_summary.json` + recalcul du sous-ensemble strict 5 langues
pour le modèle Llama fusionné (4 140 items, 5 langues). **Résultats principaux, tous sur 5 langues
strictes :**

| Modèle | Stratégie | **Cohérence `specific`** (↑ robuste) | **Diversité `unspecific`** (↑ diverse) |
|---|---|---:|---:|
| **Llama 3.1 8B (Groq)** | vanilla | **0,730** | 0,228 |
| Llama 3.1 8B (Groq) | prefix_suffix | 0,704 | 0,242 |
| **Qwen 2.5 3B** | vanilla | **0,644** | 0,329 |
| Qwen 2.5 3B | prefix_suffix | 0,587 | 0,416 |

> Robustesse de l'alignement : **4 140 items alignés sur 5 langues** pour *chaque* modèle (Llama
> fusionné comme Qwen). Le sous-ensemble strict 5-langues est donc pleinement comparable entre les
> deux modèles. Médiane de cohérence Llama 5-langues = 0,740.

**Interprétation — le résultat central du projet.** Les deux objectifs du challenge se comportent de
manière **opposée selon la taille du modèle** :

- Sur **`specific`** (le contexte est donné, on *veut* de la cohérence), **Llama 8B > Qwen 3B**
  (0,730 vs 0,644). Le gros modèle exploite mieux le contexte explicite et reste stable d'une langue
  à l'autre : **plus robuste**.
- Sur **`unspecific`** (la culture est inférée, on *veut* de la diversité), **Qwen 3B > Llama 8B**
  (0,329–0,416 vs 0,228–0,242). Le petit modèle varie davantage selon la langue : **plus divers**.

Autrement dit, le modèle le plus « fort » n'est pas le meilleur sur les deux axes : il est plus
robuste mais moins divers. C'est cohérent avec l'intuition (un modèle plus capable « normalise »
davantage ses réponses) et constitue le résultat le plus directement exploitable du Lot D.

### 5.3 Ventilation par langue (`specific`, baseline `vanilla`)

Source : `data/output/analysis/language_summary.csv`. Les agrégats du §5.1 masquent des écarts
**par langue** qui éclairent la robustesse. On reporte ici la condition `vanilla specific`, la plus
révélatrice (la variante `prefix_suffix` aplanit ces écarts, cf. §5.4–5.5). `n = 4 140` par langue
et par modèle (Qwen `fr` : 4 145).

| Modèle | Langue | mots (moy.) | % trop longues | % mentionne le pays | % mauvaise langue |
|---|---|---:|---:|---:|---:|
| Llama 3.1 8B | de | 36,9 | 33,2 | 6,0 | 0,17 |
| Llama 3.1 8B | en | 30,8 | 5,4 | 4,9 | 0,07 |
| Llama 3.1 8B | es | 34,6 | 19,3 | 17,7 | 0,27 |
| Llama 3.1 8B | fr | 33,1 | 12,1 | 2,1 | 0,10 |
| Llama 3.1 8B | it | 30,1 | 5,3 | 5,5 | 0,22 |
| Qwen 2.5 3B | de | 26,3 | 12,4 | 35,5 | 0,17 |
| Qwen 2.5 3B | en | 26,1 | 2,4 | 18,9 | 0,10 |
| Qwen 2.5 3B | es | 31,5 | 22,6 | 26,9 | **5,12** |
| Qwen 2.5 3B | fr | 27,1 | 10,8 | 23,7 | 0,36 |
| Qwen 2.5 3B | it | 22,1 | 3,9 | 9,9 | **7,13** |

**Lecture.**
- **La verbosité dépend de la langue, pas seulement du modèle.** Chez Llama, l'allemand explose la
  longueur (36,9 mots, 33,2 % trop longues) tandis que l'anglais et l'italien tiennent la phrase
  unique (≈ 5 % trop longues). L'écart « trop longues » va de **5,3 % à 33,2 %** selon la langue —
  un facteur que l'agrégat (15,1 %) gomme entièrement.
- **L'ancrage culturel diffère fortement par langue ET par modèle.** Qwen mentionne explicitement le
  pays bien plus souvent que Llama (jusqu'à **35,5 %** en allemand contre 6,0 % pour Llama). Cet
  écart d'ancrage contribue mécaniquement aux écarts de cohérence sémantique du §5.2.
- **La dérive linguistique de Qwen est concentrée sur deux langues** (it 7,13 %, es 5,12 %) ; les
  trois autres restent < 0,4 %. C'est un défaut **localisé**, pas global — détaillé au §5.6.

### 5.4 Effet de la variante — comparaison structurée baseline ↔ variante

Source : `data/output/analysis/baseline_vs_variant.csv` (items **strictement alignés** entre
baseline et variante : mêmes `id`, mêmes langues).

| Modèle | Comparaison | items alignés | % trop longues | mots (moy.) |
|---|---|---:|---:|---:|
| Llama 3.1 8B | vanilla → prefix_suffix | 12 420 | 21,5 % → **3,0 %** | 34,9 → 28,0 |
| Qwen 2.5 3B | vanilla → prefix_suffix | 20 700 | 10,5 % → **0,2 %** | 26,6 → 16,4 |

> Sur le sous-ensemble strictement aligné (mêmes `id`, mêmes langues) : Llama compte **12 420 items
> alignés** baseline↔variante (le run `prefix_suffix` couvre 3 langues × 4 140), Qwen **20 700**
> (5 langues × 4 140). Le `% trop longues` de la baseline aligné (21,5 % Llama) est plus élevé que
> l'agrégat du §5.1 (15,1 %) car il porte sur les seules langues présentes dans la variante.

Croisé avec §5.2, l'effet de `prefix_suffix` est net : **forte discipline de forme** (longueur,
concision, langue cible) pour un **effet de fond modeste et plutôt dans le sens d'une légère hausse
de la diversité** (cohérence specific Llama 0,730 → 0,704 ; diversité unspecific Qwen 0,329 → 0,416).
La contrainte de style n'homogénéise pas les contenus entre langues — au contraire, en raccourcissant
les réponses, elle laisse moins de place au « tronc commun » générique partagé entre langues.

### 5.5 Généricité sémantique (mesure par embeddings)

Source : notebook `analyse_lot_d_final.ipynb`, section 7. Cette mesure complète §5.2 : au lieu de
comparer les langues entre elles, elle situe **chaque réponse par rapport au centroïde de sa propre
condition**. La `semantic_specificity` d'une réponse = `1 − cosinus(réponse, centroïde de la
condition)`. **↑ = réponse plus singulière** (s'éloigne du « réponse-type » de la condition) ;
**↓ = réponse plus générique** (proche du passe-partout).

| Modèle | Stratégie | Dataset | **`semantic_specificity` moy.** (↑ singulier) |
|---|---|---|---:|
| **Qwen 2.5 3B** | prefix_suffix | specific | **0,606** |
| Qwen 2.5 3B | prefix_suffix | unspecific | 0,594 |
| Llama 3.1 8B | vanilla | specific | 0,566 |
| Qwen 2.5 3B | vanilla | specific | 0,565 |
| Qwen 2.5 3B | vanilla | unspecific | 0,564 |
| Llama 3.1 8B | prefix_suffix | unspecific | 0,562 |
| Llama 3.1 8B | vanilla | unspecific | 0,560 |
| Llama 3.1 8B | prefix_suffix | specific | 0,560 |

**Lecture.**
- **C'est une mesure de fond (embeddings), pas lexicale** — contrairement aux heuristiques de
  mots-clés du §6, qu'elle vient compléter. Elle dit à quel point les réponses d'une condition se
  ressemblent *entre elles*.
- **Contre-intuitivement, Qwen `prefix_suffix` est le plus singulier** (0,606 sur `specific`), bien
  qu'il produise les réponses les plus courtes (≈ 16 mots). Concision n'implique donc pas généricité :
  en forçant la langue cible et un format serré, l'enrobage écarte les réponses du tronc commun
  multilingue anglophone plutôt que de les y ramener — cohérent avec la hausse de diversité observée
  au §5.2.
- **Les écarts restent resserrés (0,560–0,606).** L'interprétation doit donc rester prudente : aucune
  condition ne s'effondre en généricité.

> **Honnêteté méthodologique — pourquoi nous ne reportons PAS `genericity_risk`.** Le notebook
> calcule aussi un drapeau `genericity_risk` (réponses sous le 25ᵉ percentile de
> `semantic_specificity`). Mais ce seuil étant fixé **par condition**, la part de réponses flaggées
> vaut **≈ 25 % par construction** dans *toutes* les conditions (24,9–25,2 % mesurés) : c'est un
> artefact de définition, **pas un résultat comparatif**. Nous reportons donc uniquement la moyenne
> continue `semantic_specificity`, seule grandeur comparable entre conditions.

### 5.6 Langue de réponse (robustesse linguistique)

Source : `data/output/analysis/lang_mismatch.csv`. Question du sujet : « quand on demande en
français, répond-il en français ? » Détection par `langdetect` (graine fixée, déterministe).

Le taux de réponses dans la **mauvaise langue** est quasi nul partout, **sauf Qwen 2.5 3B vanilla** :

| Qwen 2.5 3B vanilla `specific` | % mauvaise langue |
|---|---:|
| it | **7,1 %** |
| es | **5,1 %** |
| fr | 0,4 % |
| de | 0,2 % |
| en | 0,1 % |

Qwen vanilla dérive vers une autre langue (souvent l'anglais) sur l'italien et l'espagnol. La
variante **`prefix_suffix` corrige ce défaut** (Qwen specific : it 7,1 % → 0,2 %, es 5,1 % → 1,7 %)
en imposant explicitement la langue cible dans l'enrobage. Llama ne présente pas ce problème
(≤ 0,27 %).

---

## 6. Analyse qualitative — typologie d'écarts

Au-delà des chiffres, le notebook (section 9) extrait des exemples de cas extrêmes (cohérence la plus
basse, diversité la plus haute, réponses trop courtes/longues). On en tire la typologie suivante,
illustrée par les métriques de §5 :

1. **Non-respect de la consigne linguistique.** Le cas le plus net : Qwen vanilla répond en anglais à
   des questions italiennes/espagnoles (§5.6). Cause probable : un petit modèle multilingue dont
   l'« attracteur » par défaut est l'anglais quand la consigne de langue n'est pas explicitée.

2. **Généricité (réponses passe-partout).** Mesurée sémantiquement par la distance au centroïde de la
   condition (`semantic_specificity`, chiffrée au §5.5). Les écarts entre conditions sont resserrés
   (0,560–0,606) : aucune condition ne s'effondre en généricité. Point notable et **contre-intuitif**,
   Qwen `prefix_suffix` est la condition la *plus* singulière (0,606) malgré ses réponses très courtes
   (≈ 16 mots) — la concision **ne se paie donc pas** mécaniquement en généricité ici.

3. **Verbosité / hors-format.** Llama vanilla dépasse souvent la « phrase unique » attendue (15,1 % en
   `specific`, 24,2 % en `unspecific`), avec un fort effet de langue (de : 33,2 % ; en/it : ≈ 5 %, cf.
   §5.3). C'est un écart de format, pas de fond, entièrement corrigé par la variante.

4. **Mention explicite du pays.** `mentions_country_pct` varie fortement (Qwen vanilla `de` : 35,5 %
   vs Llama `de` : 6,0 % — cf. §5.3) : les modèles n'ancrent pas leurs réponses dans le contexte avec
   la même intensité, ce qui contribue aux écarts de cohérence observés.

> **Honnêteté méthodologique.** Les colonnes `vague_kw_pct` et `stereotype_kw_pct` sont des
> **heuristiques lexicales** (comptage de mots-clés type « toujours / always / typically »). Elles
> sont indicatives et **explicitement étiquetées comme non sémantiques** dans le notebook et ici. La
> seule mesure de fond fiable reste l'embedding (§5.2–5.3 et §5.5). Nous ne présentons donc pas le comptage
> de mots-clés comme une « détection de stéréotypes ».

---

## 7. Limites

1. **Couverture des variantes.** Seules `vanilla` et `prefix_suffix` (C2) disposent de runs complets
   exploités. C1 (`system_prompt`) et C3 (`rewrite`) sont validés fonctionnellement mais sans run
   complet committé ; leurs effets de fond ne sont donc pas chiffrés ici.
2. **Volet qualitatif partiellement lexical.** La généricité est chiffrée par embeddings (§5.5), mais
   les colonnes « vague » / « stéréotype » restent des heuristiques de mots-clés (cf. §6). Une vraie
   détection de stéréotypes / hallucinations culturelles demanderait un LLM-juge (piste §8).
3. **Deux modèles, une famille de tailles.** La comparaison « gros vs petit » repose sur deux modèles
   d'architectures différentes (Llama vs Qwen) ; l'effet « taille » et l'effet « famille » ne sont
   pas séparables. Conclusion à lire comme « ces deux systèmes » plutôt que « la taille en général ».
4. **`unspecific` à faible volume.** 505 réponses par condition (101 par langue) contre des milliers
   en `specific` : les chiffres de diversité sont solides en tendance mais moins finement résolus.

---

## 8. Recommandations

- **Pour la robustesse (`specific`)** : privilégier **Llama 3.1 8B via Groq** (cohérence la plus
  élevée, latence cloud ~165 ms, taux de mauvaise langue ~0 %). C'est le meilleur candidat pour une
  soumission « robustesse » à grand volume.
- **Pour la diversité (`unspecific`)** : **Qwen 2.5 3B** produit plus de variation inter-langues ;
  intéressant pour explorer la diversité culturelle, au prix d'un risque de dérive linguistique à
  corriger par un enrobage `prefix_suffix`.
- **En production** : ajouter une variante `prefix_suffix` systématique pour garantir langue + format
  sans coût d'inférence supplémentaire (1 seul appel).
- **Pistes d'extension** : (1) lancer les runs complets C1/C3 pour chiffrer leur effet de fond ;
  (2) ajouter un **LLM-juge** (réutilisant `GroqProvider`) pour scorer pertinence/cohérence et
  qualifier vraiment les stéréotypes ; (3) produire l'**export au format challenge**
  (`submission_metadata.json` conforme au schéma `team / system / model / submissionid / date /
  label / languages / modifications{…}`) pour une soumission officielle.

---

## 9. Reproductibilité

Un tiers doit pouvoir relancer l'analyse :

```powershell
# 1. Environnement
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -e ".[analysis]"

# 2. Vérifier les dépendances d'analyse
python -c "import numpy, pandas, sklearn, sentence_transformers, seaborn, langdetect; print('Lot D deps OK')"

# 3. Rejouer l'analyse de bout en bout
jupyter notebook notebooks/analyse_lot_d_final.ipynb   # puis Kernel > Restart & Run All
```

**Artefacts versionnés** (visibles sans relancer le notebook) sous `data/output/analysis/` :

| Fichier | Contenu |
|---|---|
| `run_summary.csv` | Stats simples par condition (longueur, % vides, % trop longues, mauvaise langue) |
| `language_summary.csv` | Mêmes stats, ventilées par langue |
| `specific_coherence.csv` | Cohérence par item aligné (avec `n_languages`) |
| `unspecific_diversity.csv` | Diversité par item aligné |
| `baseline_vs_variant.csv` | Comparaison structurée baseline ↔ variante (items alignés) |
| `lang_mismatch.csv` | % de réponses dans la mauvaise langue, par condition × langue |
| `analysis_summary.json` | Résumé machine-lisible (76 545 réponses) |
| `figures/*.png` | Métriques par langue + cohérence/diversité |

**Runs et configs** : `configs/*.yaml` (baseline + variantes), `data/output/runs/*/`
(sorties JSONL + `config_snapshot.yaml` + `run_metadata.json` par run).

---

*Rapport final du Lot E. Les sorties JSONL sont conformes (UTF-8, un objet par ligne, champ `answer`
et `prompt_trace` ajoutés). Reste à produire le builder de métadonnées au format challenge pour une
soumission officielle (cf. §8).*
