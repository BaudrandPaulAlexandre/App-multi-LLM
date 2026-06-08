import gradio as gr
import requests
import time
import os
import tempfile

# Pointez sur le Mock Server (8000) pour développer, puis passez à 8000 pour la prod.
API_BASE_URL = "http://localhost:8000"

def fetch_catalogue():
    """Récupère les providers, langues et stratégies depuis le backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/providers", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Erreur de connexion au backend: {e}")
        return {"providers": {}, "languages": [], "strategies": []}

CATALOGUE = fetch_catalogue()

# --- Fonctions d'interaction avec l'API ---

def update_models(provider_id):
    """Met à jour la liste des modèles quand on change de provider."""
    if not provider_id or provider_id not in CATALOGUE.get("providers", {}):
        return gr.update(choices=[], value=None)
    models = [m["id"] for m in CATALOGUE["providers"][provider_id]["models"]]
    return gr.update(choices=models, value=models[0] if models else None)

def launch_and_track_run(provider, model, langs, dataset_type, temp, max_tokens, max_questions, strategy):
    """
    Lance un run (POST) puis fait du polling (GET /status) pour 
    mettre à jour l'interface en temps réel (Générateur Gradio).
    """
    if not langs:
        yield "❌ Erreur : Veuillez sélectionner au moins une langue.", gr.update(visible=False)
        return

    payload = {
        "provider": provider,
        "model": model,
        "languages": langs,
        "dataset_type": dataset_type,
        "temperature": temp,
        "max_tokens": max_tokens,
        "max_questions": max_questions,
        "strategy": strategy
    }

    try:
        # 1. Lancer le run
        resp = requests.post(f"{API_BASE_URL}/runs", json=payload)
        resp.raise_for_status()
        run_data = resp.json()
        run_id = run_data["run_id"]
        
        # 2. Polling (boucle de suivi)
        status = "started"
        while status not in ["done", "error"]:
            time.sleep(2) # Polling toutes les 2 secondes
            status_resp = requests.get(f"{API_BASE_URL}/runs/{run_id}/status")
            if status_resp.status_code == 200:
                s_data = status_resp.json()
                status = s_data.get("status", "error")
                done = s_data.get("questions_done", 0)
                total = s_data.get("questions_total", 1)
                lang = s_data.get("current_language", "")
                
                progress_text = f"⏳ En cours... Run ID: {run_id}\nLangue actuelle: {lang} | Progression: {done}/{total} questions"
                yield progress_text, gr.update(visible=False)
            else:
                break
                
        # 3. Fin du run
        if status == "done":
            zip_url  = f"{API_BASE_URL}/runs/{run_id}/download"
            yaml_url = f"{API_BASE_URL}/runs/{run_id}/config.yaml"
            # On génère directement les balises HTML pour les boutons de téléchargement
            html_btn = (
                f'<a href="{zip_url}" target="_blank" style="display:inline-block; margin:4px 8px 4px 0; padding:10px 15px; background-color:#22c55e; color:white; font-weight:bold; border-radius:5px; text-decoration:none;">📥 Télécharger le package (.zip)</a>'
                f'<a href="{yaml_url}" target="_blank" style="display:inline-block; margin:4px 0; padding:10px 15px; background-color:#3b82f6; color:white; font-weight:bold; border-radius:5px; text-decoration:none;">📄 Télécharger le YAML du run</a>'
            )
            yield f"✅ Terminé avec succès ! Run ID: {run_id}", gr.update(value=html_btn, visible=True)
        else:
            yield f"❌ Erreur lors du run {run_id}", gr.update(visible=False)
            
    except requests.exceptions.RequestException as e:
        yield f"❌ Erreur réseau : {str(e)}", gr.update(visible=False)

def get_history():
    """Récupère l'historique des runs pour l'onglet correspondant."""
    try:
        resp = requests.get(f"{API_BASE_URL}/runs")
        resp.raise_for_status()
        runs = resp.json()
        
        # Formatage pour un affichage en tableau
        formatted = []
        for r in runs:
            formatted.append([
                r.get("run_id"), r.get("status"), r.get("provider"), 
                r.get("model"), ", ".join(r.get("languages", [])), 
                f"{round(r.get('duration_seconds', 0), 2)}s" if r.get('duration_seconds') else "-"
            ])
        return formatted
    except Exception:
        return [["Erreur", "Impossible de charger l'historique", "", "", "", ""]]

def generate_yaml(provider, model, langs, dataset_type, temp, max_tokens, max_questions, strategy):
    """
    Génère un fichier YAML de configuration depuis les choix du formulaire
    (sans lancer de run) et le propose au téléchargement.
    """
    if not langs:
        return gr.update(visible=False), "❌ Sélectionnez au moins une langue avant de générer le YAML."

    payload = {
        "provider": provider,
        "model": model,
        "languages": langs,
        "dataset_type": dataset_type,
        "temperature": temp,
        "max_tokens": max_tokens,
        "max_questions": max_questions,
        "strategy": strategy,
    }

    try:
        resp = requests.post(f"{API_BASE_URL}/config/yaml", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        detail = str(e)
        if getattr(e, "response", None) is not None:
            try:
                detail = e.response.json().get("detail", detail)
            except Exception:
                pass
        return gr.update(visible=False), f"❌ Impossible de générer le YAML : {detail}"

    # Écrit le YAML dans un fichier temporaire pour le composant de téléchargement
    tmp_dir = tempfile.mkdtemp(prefix="eloquent_yaml_")
    file_path = os.path.join(tmp_dir, data["filename"])
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(data["yaml"])

    return gr.update(value=file_path, visible=True), f"✅ YAML généré : `{data['filename']}` — réutilisable avec `python run.py --config <fichier>`"

def history_yaml_link(run_id):
    """Construit un lien de téléchargement du config_snapshot.yaml d'un run passé."""
    run_id = (run_id or "").strip()
    if not run_id:
        return gr.update(value="<i>Entrez un Run ID (colonne de gauche du tableau ci-dessus).</i>", visible=True)
    url = f"{API_BASE_URL}/runs/{run_id}/config.yaml"
    link = (
        f'<a href="{url}" target="_blank" style="display:inline-block; padding:8px 14px; '
        f'background-color:#3b82f6; color:white; font-weight:bold; border-radius:5px; '
        f'text-decoration:none;">📄 Télécharger {run_id}_config.yaml</a>'
    )
    return gr.update(value=link, visible=True)

# --- Interface Utilisateur (Gradio Blocks) ---

with gr.Blocks(title="ELOQUENT - Panel de Contrôle (Lot B)") as app:
    gr.Markdown("# 🌍 Challenge ELOQUENT - Interface Multi-LLM")
    
    with gr.Tabs():
        # --- ONGLET 1 : LANCEMENT ---
        with gr.Tab("🚀 Lancer une Expérience"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚙️ Configuration du Modèle")
                    
                    # Fix 422 : Initialisation correcte avec des valeurs par défaut
                    provider_list = list(CATALOGUE.get("providers", {}).keys())
                    default_provider = provider_list[0] if provider_list else None
                    default_models = [m["id"] for m in CATALOGUE["providers"][default_provider]["models"]] if default_provider else []
                    
                    provider_dropdown = gr.Dropdown(choices=provider_list, label="Provider", value=default_provider)
                    model_dropdown = gr.Dropdown(choices=default_models, label="Modèle", value=default_models[0] if default_models else None)
                    
                    provider_dropdown.change(fn=update_models, inputs=provider_dropdown, outputs=model_dropdown)
                    
                    dataset_type = gr.Radio(choices=["specific", "unspecific"], label="Type de Dataset", value="specific")
                    
                    lang_choices = [l["code"] for l in CATALOGUE.get("languages", [])]
                    languages = gr.CheckboxGroup(choices=lang_choices, label="Langues", value=["fr"])

                    # Sélecteur de stratégie de prompting (Lot A + Lot C) — peuplé
                    # dynamiquement depuis le backend (GET /providers).
                    strat_pairs = [(s["label"], s["id"]) for s in CATALOGUE.get("strategies", [])]
                    strat_ids   = [s["id"] for s in CATALOGUE.get("strategies", [])]
                    strategy = gr.Dropdown(
                        choices=strat_pairs,
                        label="Stratégie de prompting",
                        value="vanilla" if "vanilla" in strat_ids else (strat_ids[0] if strat_ids else None),
                        info="vanilla = baseline (Lot A) · les autres sont les variantes du Lot C",
                    )

            with gr.Row():
                launch_btn = gr.Button("▶️ Lancer le Run", variant="primary", scale=2)
                yaml_btn   = gr.Button("📄 Générer le YAML de configuration", scale=1)

            gr.Markdown("### 📊 Progression en direct")
            status_box = gr.Textbox(label="Statut", interactive=False)
            download_html = gr.HTML(visible=False, label="Téléchargement")

            gr.Markdown("### 📄 Configuration YAML (sans lancer de run)")
            yaml_status = gr.Markdown(visible=True)
            yaml_file   = gr.File(label="Fichier YAML généré", visible=False)
            
        # --- ONGLET 2 : HISTORIQUE ---
        with gr.Tab("📜 Historique des Runs"):
            refresh_btn = gr.Button("🔄 Rafraîchir l'historique")
            history_table = gr.Dataframe(
                headers=["Run ID", "Statut", "Provider", "Modèle", "Langues", "Durée"],
                interactive=False
            )
            refresh_btn.click(fn=get_history, inputs=[], outputs=history_table)
            app.load(fn=get_history, inputs=[], outputs=history_table) # Charge au démarrage

            gr.Markdown("### 📄 Télécharger le YAML d'un run passé")
            with gr.Row():
                hist_run_id   = gr.Textbox(label="Run ID", placeholder="ex : groq_vanilla_20260607_174010", scale=3)
                hist_yaml_btn = gr.Button("📄 Obtenir le lien YAML", scale=1)
            hist_yaml_link = gr.HTML(visible=False)
            hist_yaml_btn.click(fn=history_yaml_link, inputs=hist_run_id, outputs=hist_yaml_link)

        # --- ONGLET 3 : PARAMÈTRES DE GÉNÉRATION ---
        with gr.Tab("🎛️ Paramètres de Génération"):
            temperature = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=0.0, label="Température (0 = Baseline déterministe)")
            max_tokens = gr.Slider(minimum=10, maximum=500, step=10, value=150, label="Max Tokens (Réponse courte)")
            max_questions = gr.Slider(minimum=5, maximum=4140, step=5, value=5, label="Max Questions")

    # Action du bouton de lancement : on relie les composants des onglets 1 et 3
    launch_btn.click(
        fn=launch_and_track_run,
        inputs=[provider_dropdown, model_dropdown, languages, dataset_type, temperature, max_tokens, max_questions, strategy],
        outputs=[status_box, download_html]
    )

    # Action du bouton "Générer le YAML" : mêmes entrées, mais produit un fichier
    # téléchargeable au lieu de lancer un run.
    yaml_btn.click(
        fn=generate_yaml,
        inputs=[provider_dropdown, model_dropdown, languages, dataset_type, temperature, max_tokens, max_questions, strategy],
        outputs=[yaml_file, yaml_status]
    )
            
# Lancement de l'application
if __name__ == "__main__":
    app.launch()