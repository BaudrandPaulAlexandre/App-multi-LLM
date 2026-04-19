import gradio as gr
import requests
import time
import os

# Pointez sur le Mock Server (8001) pour développer, puis passez à 8000 pour la prod.
API_BASE_URL = "http://localhost:8001"

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
    if not provider_id or provider_id not in CATALOGUE["providers"]:
        return gr.update(choices=[], value=None)
    models = [m["id"] for m in CATALOGUE["providers"][provider_id]["models"]]
    return gr.update(choices=models, value=models[0] if models else None)

def launch_and_track_run(provider, model, langs, dataset_type, temp, max_tokens, strategy):
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
            download_url = f"{API_BASE_URL}/runs/{run_id}/download"
            yield f"✅ Terminé avec succès ! Run ID: {run_id}", gr.update(value=download_url, visible=True)
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
                f"{r.get('duration_seconds', 0)}s" if r.get('duration_seconds') else "-"
            ])
        return formatted
    except Exception:
        return [["Erreur", "Impossible de charger l'historique", "", "", "", ""]]

# --- Interface Utilisateur (Gradio Blocks) ---

with gr.Blocks(title="ELOQUENT - Panel de Contrôle (Lot B)") as app:
    gr.Markdown("# 🌍 Challenge ELOQUENT - Interface Multi-LLM")
    
    with gr.Tabs():
        # --- ONGLET 1 : LANCEMENT ---
        with gr.Tab("🚀 Lancer une Expérience"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚙️ Configuration du Modèle")
                    provider_list = list(CATALOGUE["providers"].keys())
                    provider_dropdown = gr.Dropdown(choices=provider_list, label="Provider", value=provider_list[0] if provider_list else None)
                    model_dropdown = gr.Dropdown(label="Modèle")
                    provider_dropdown.change(fn=update_models, inputs=provider_dropdown, outputs=model_dropdown)
                    
                    dataset_type = gr.Radio(choices=["specific", "unspecific"], label="Type de Dataset", value="specific")
                    
                    lang_choices = [l["code"] for l in CATALOGUE["languages"]]
                    languages = gr.CheckboxGroup(choices=lang_choices, label="Langues", value=["fr"])
                
                with gr.Column():
                    gr.Markdown("### 🎛️ Paramètres de Génération")
                    temperature = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=0.0, label="Température (0 = Baseline déterministe)")
                    max_tokens = gr.Slider(minimum=10, maximum=500, step=10, value=150, label="Max Tokens (Réponse courte)")
                    
                    strat_choices = [s["id"] for s in CATALOGUE["strategies"]]
                    strategy = gr.Dropdown(choices=strat_choices, label="Stratégie (Lot C)", value="vanilla")
                    
            launch_btn = gr.Button("▶️ Lancer le Run", variant="primary")
            
            gr.Markdown("### 📊 Progression en direct")
            status_box = gr.Textbox(label="Statut", interactive=False)
            # Bouton de téléchargement masqué par défaut, on passera l'URL en javascript ou html
            download_html = gr.HTML(visible=False, label="Téléchargement")
            
            # Action du bouton de lancement (utilise un générateur pour la barre de progression texte)
            launch_btn.click(
                fn=launch_and_track_run,
                inputs=[provider_dropdown, model_dropdown, languages, dataset_type, temperature, max_tokens, strategy],
                outputs=[status_box, download_html]
            )
            # Met à jour le bouton de téléchargement avec un vrai lien cliquable quand c'est prêt
            download_html.change(fn=lambda url: f'<a href="{url}" target="_blank" style="padding:10px; background-color:#22c55e; color:white; border-radius:5px; text-decoration:none;">📥 Télécharger le package de soumission (.zip)</a>', inputs=download_html, outputs=download_html)
            
        # --- ONGLET 2 : HISTORIQUE ---
        with gr.Tab("📜 Historique des Runs"):
            refresh_btn = gr.Button("🔄 Rafraîchir l'historique")
            history_table = gr.Dataframe(
                headers=["Run ID", "Statut", "Provider", "Modèle", "Langues", "Durée"],
                interactive=False
            )
            refresh_btn.click(fn=get_history, inputs=[], outputs=history_table)
            app.load(fn=get_history, inputs=[], outputs=history_table) # Charge au démarrage

# Lancement de l'application
if __name__ == "__main__":
    app.launch()