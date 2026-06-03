# App-multi-LLM : Challenge ELOQUENT (Cultural Robustness & Diversity)

## Contexte du Projet
Ce projet s'inscrit dans le cadre du module **Big Data Analytics** et vise à répondre au challenge **ELOQUENT @ CLEF 2026**. L'objectif est de développer un pipeline d'évaluation multi-LLM permettant de mesurer la robustesse et la diversité culturelle des modèles de langage génératifs.

Deux dimensions sont particulièrement étudiées :
* **Cultural Diversity (unspecific) :** La culture est inférée par la langue de la question. On s'attend à des réponses différentes selon le prisme linguistique.
* **Cultural Robustness (specific) :** Le contexte culturel (ou le pays) est explicitement fourni dans le prompt. On s'attend à des réponses cohérentes entre les langues.

## Architecture Technique
L'application est conçue pour être modulaire et robuste, séparant clairement la logique métier de l'interface utilisateur :

* **Backend (FastAPI) :** Un serveur REST qui orchestre les requêtes vers les fournisseurs de modèles (API externe Groq ou Ollama en local). Il gère la validation stricte des données via Pydantic et le suivi asynchrone des exécutions (runs).
* **Frontend (Gradio) :** Une interface utilisateur interactive (Lot B) offrant un panneau de contrôle pour configurer les expériences, suivre la progression de la génération en temps réel, consulter l'historique complet et télécharger les paquets de résultats formatés pour la soumission.

## Fonctionnalités Principales
* **Interface Unifiée :** Lancement de benchmarks simultanés sur un panel de 5 langues (fr, en, es, it, de).
* **Multi-Providers :** Intégration fluide de différents modèles, des plus légers en local (ex: Qwen 2.5 via Ollama) aux modèles hébergés sur le cloud (ex: Llama 3.1 via Groq).
* **Stratégies de Prompting (Lot C) :** Application de différentes variantes à la volée pour forcer la concision ou améliorer la cohérence :
    * *Vanilla :* Génération brute (Baseline).
    * *System Prompt :* Injection de contraintes de concision au niveau système.
    * *Prefix/Suffix :* Ajout de marqueurs linguistiques stricts dans la langue cible.
    * *Rewrite :* Pipeline en deux passes corrigeant les réponses trop longues.

---

## Prérequis & Installation

### 1. Prérequis Système
* Python 3.10+
* Une clé API Groq (pour l'accès aux modèles Llama)
* Ollama installé et lancé en local (uniquement si vous testez les modèles locaux)

### 2. Configuration de l'environnement
Clonez le dépôt, puis ouvrez un terminal à la racine du projet. Créez et activez un environnement virtuel isolé :

```powershell
# Création de l'environnement
python -m venv .venv

# Activation de l'environnement :
.\.venv\Scripts\Activate.ps1  

#Si autorisation nécessaire, utiliser avant : 
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Installez l'ensemble des dépendances du projet (FastAPI, Gradio, Uvicorn, etc.) via le fichier de configuration :

```powerShell
pip install -e .

# Configurez vos variables d'environnement en créant un fichier nommé exactement .env à la racine du projet :
# On peut y trouver par exemple : 

GROQ_API_KEY=gsk_votre_cle_api_secrete_ici

# Utilisation
# L'architecture nécessite de lancer le serveur et l'interface séparément. Ouvrez deux terminaux distincts et assurez-vous que votre environnement virtuel (.venv) est activé dans les deux.

# Terminal 1 : Démarrer le Serveur Backend

```powerShell
uvicorn server:app --reload --host 0.0.0.0 --port 8000
# Le serveur écoute désormais les requêtes et l'API est documentée automatiquement sur http://localhost:8000/docs.

# Terminal 2 : Démarrer l'Interface Frontend

```powerShell
python app_gradio.py
# L'interface graphique est accessible depuis votre navigateur à l'adresse indiquée dans le terminal (généralement http://127.0.0.1:7860). Vous pouvez configurer vos modèles, lancer vos runs et télécharger vos fichiers JSONL générés.

# Structure du Dépôt

data/ : Contient les jeux de données d'entrée (input) et les résultats générés (output/runs).
|
src/eloquent/ : Le cœur logique de l'application (pipeline, providers, stratégies de prompting).
|
configs/ : Fichiers YAML définissant les paramètres stricts de chaque variante expérimentale.
|
server.py : Point d'entrée de l'API FastAPI.
|
app_gradio.py : Point d'entrée de l'interface utilisateur.
