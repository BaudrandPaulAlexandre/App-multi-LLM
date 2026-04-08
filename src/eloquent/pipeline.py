"""
pipeline.py
-----------
PipelineRunner : boucle principale du Lot A.

Lit les fichiers JSONL d'entrée (un par langue), interroge le LLM
pour chaque question, écrit les fichiers JSONL de sortie avec le
champ "answer" ajouté.

Usage via run.py :
    runner = PipelineRunner(config)
    runner.run()
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from eloquent.config import RunConfig
from eloquent.logger import get_logger
from eloquent.prompting import build_strategy
from eloquent.providers import LLMProvider, LLMResponse, build_provider_from_config

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers JSONL
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    """Lit un fichier JSONL et retourne une liste de dicts."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Ligne %d ignorée (JSON invalide) : %s", line_num, exc)
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    """Écrit une liste de dicts dans un fichier JSONL (une ligne par dict)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_question_field(record: dict) -> str | None:
    """
    Détecte le champ contenant la question dans un enregistrement JSONL.
    Le challenge ELOQUENT utilise différents noms selon les fichiers.
    Ordre de priorité : "question", "query", "text", premier champ string trouvé.
    """
    for field in ("question", "query", "text", "prompt"):
        if field in record and isinstance(record[field], str):
            return field
    # Fallback : premier champ dont la valeur est une string non vide
    for key, val in record.items():
        if isinstance(val, str) and val.strip():
            return key
    return None


# ---------------------------------------------------------------------------
# PipelineRunner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """
    Orchestre l'exécution d'un run ELOQUENT complet.

    Flux :
        1. Crée le dossier de run horodaté
        2. Sauvegarde un snapshot de la config
        3. Pour chaque langue × dataset_type :
            a. Lit le fichier JSONL d'entrée
            b. Pour chaque question → appelle le LLM → ajoute "answer"
            c. Écrit le fichier JSONL de sortie
        4. Écrit les métadonnées du run (stats, erreurs, durée)
    """

    def __init__(self, config: RunConfig) -> None:
        self.cfg = config
        self.provider: LLMProvider = build_provider_from_config(config)
        self.strategy = build_strategy(config.prompting.strategy)

        # Dossier de run : output_dir / run_id_YYYYMMDD_HHMMSS
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_dir = (
            config.paths.output_dir / f"{config.run_id}_{timestamp}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Dossier de run : %s", self.run_dir)

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Lance le run complet.
        Retourne un dict de métadonnées (aussi sauvegardé dans run_metadata.json).
        """
        start_time = datetime.now(timezone.utc)
        logger.info(
            "Démarrage du run '%s' | provider=%s | stratégie=%s",
            self.cfg.run_id,
            self.cfg.provider,
            self.cfg.prompting.strategy,
        )

        # Sauvegarde du snapshot de config dans le dossier de run
        self._save_config_snapshot()

        # Vérification que le provider répond avant de lancer
        if not self.provider.health_check():
            raise RuntimeError(
                f"Le provider '{self.cfg.provider}' est inaccessible. "
                f"Vérifiez votre connexion / qu'Ollama tourne."
            )

        # Boucle sur les langues
        run_stats: dict[str, dict] = {}
        for lang in self.cfg.languages:
            lang_stats = self._process_language(lang)
            run_stats[lang] = lang_stats

        end_time = datetime.now(timezone.utc)
        duration_s = (end_time - start_time).total_seconds()

        metadata = {
            "run_id": self.cfg.run_id,
            "provider": self.cfg.provider,
            "model": self.cfg.model,
            "strategy": self.cfg.prompting.strategy,
            "dataset_type": self.cfg.dataset_type,
            "languages": self.cfg.languages,
            "generation": self.cfg.generation.to_dict(),
            "started_at": start_time.isoformat(),
            "ended_at": end_time.isoformat(),
            "duration_seconds": round(duration_s, 1),
            "per_language": run_stats,
        }

        self._save_metadata(metadata)

        total_ok = sum(s["success"] for s in run_stats.values())
        total_err = sum(s["errors"] for s in run_stats.values())
        logger.info(
            "Run terminé en %.1fs — %d réponses OK, %d erreurs",
            duration_s, total_ok, total_err,
        )
        return metadata

    # ------------------------------------------------------------------
    # Traitement d'une langue
    # ------------------------------------------------------------------

    def _process_language(self, lang: str) -> dict:
        """
        Traite tous les fichiers JSONL pour une langue donnée.
        Retourne des stats (nb questions, succès, erreurs, latence moyenne).
        """
        input_path = (
            self.cfg.paths.input_dir
            / f"{lang}_{self.cfg.dataset_type}.jsonl"
        )

        if not input_path.exists():
            logger.warning(
                "Fichier introuvable, langue ignorée : %s", input_path
            )
            return {"success": 0, "errors": 0, "skipped": True}

        records = read_jsonl(input_path)
        logger.info(
            "[%s] %d questions trouvées dans %s",
            lang, len(records), input_path.name,
        )

        output_records = []
        success_count = 0
        error_count = 0
        total_latency_ms = 0.0

        for record in tqdm(records, desc=f"{lang}", unit="q", leave=False):
            processed, resp = self._process_record(record, lang)
            output_records.append(processed)

            if resp.success:
                success_count += 1
                total_latency_ms += resp.latency_ms
            else:
                error_count += 1
                # Log l'erreur avec l'ID de la question pour faciliter le debug
                q_id = record.get("id", record.get("query_id", "?"))
                logger.error(
                    "[%s] Erreur sur question id=%s : %s",
                    lang, q_id, resp.error,
                )

        # Écriture du fichier de sortie
        output_path = (
            self.run_dir / f"{lang}_{self.cfg.dataset_type}_output.jsonl"
        )
        write_jsonl(output_records, output_path)
        logger.info("[%s] Sortie écrite : %s", lang, output_path.name)

        avg_latency = (
            total_latency_ms / success_count if success_count > 0 else 0.0
        )

        return {
            "total": len(records),
            "success": success_count,
            "errors": error_count,
            "avg_latency_ms": round(avg_latency, 1),
        }

    # ------------------------------------------------------------------
    # Traitement d'une question
    # ------------------------------------------------------------------

    def _process_record(
        self, record: dict, lang: str
    ) -> tuple[dict, LLMResponse]:
        """
        Traite une question unique :
          1. Détecte le champ contenant la question
          2. Construit les messages via la stratégie de prompting
          3. Appelle le provider (generate_safe = pas d'exception)
          4. Ajoute "answer" au record

        Retourne le record enrichi + la LLMResponse (pour les stats).
        """
        question_field = find_question_field(record)

        if question_field is None:
            logger.warning(
                "[%s] Aucun champ question trouvé dans le record : %s",
                lang, list(record.keys()),
            )
            dummy_resp = LLMResponse(
                content="",
                model=self.cfg.model,
                provider_name=self.cfg.provider,
                latency_ms=0.0,
                error="Champ question introuvable",
            )
            return {**record, "answer": ""}, dummy_resp

        question_text = record[question_field]
        messages = self.strategy.build_messages(question_text)

        resp = self.provider.generate_safe(
            messages=messages,
            temperature=self.cfg.generation.temperature,
            max_tokens=self.cfg.generation.max_tokens,
        )

        # On enrichit le record original sans le modifier (dict unpacking)
        enriched = {**record, "answer": resp.content}
        return enriched, resp

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def _save_config_snapshot(self) -> None:
        """Sauvegarde la config utilisée dans le dossier de run."""
        snapshot_path = self.run_dir / "config_snapshot.yaml"
        import yaml
        with snapshot_path.open("w", encoding="utf-8") as f:
            yaml.dump(self.cfg.to_dict(), f, allow_unicode=True, sort_keys=False)
        logger.info("Snapshot config sauvegardé : %s", snapshot_path.name)

    def _save_metadata(self, metadata: dict) -> None:
        """Sauvegarde les métadonnées du run en JSON."""
        meta_path = self.run_dir / "run_metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info("Métadonnées sauvegardées : %s", meta_path.name)


# ---------------------------------------------------------------------------
# Mise à jour du fichier progress.json (lu par le Lot B en temps réel)
# ---------------------------------------------------------------------------

    def _write_progress(
        self,
        status: str,
        current_language: str = "",
        languages_done: list[str] | None = None,
        questions_done: int = 0,
        questions_total: int = 0,
        errors_count: int = 0,
        last_error: str = "",
    ) -> None:
        """
        Écrit / met à jour progress.json dans le dossier de run.
        Appelé régulièrement pendant _process_language() pour que
        le Lot B puisse afficher la progression en temps réel.
        """
        progress = {
            "run_id":             self.cfg.run_id,
            "status":             status,
            "current_language":   current_language,
            "languages_done":     languages_done or [],
            "languages_total":    self.cfg.languages,
            "questions_done":     questions_done,
            "questions_total":    questions_total,
            "errors_count":       errors_count,
            "last_error":         last_error,
            "started_at":         self._started_at,
            "updated_at":         datetime.now(timezone.utc).isoformat(),
        }
        progress_path = self.run_dir / "progress.json"
        with progress_path.open("w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
