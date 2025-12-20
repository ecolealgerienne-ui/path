#!/usr/bin/env python3
"""
Active Learning - Collecte de Feedback Expert.

Ce module permet aux pathologistes de signaler leurs désaccords
avec les prédictions du modèle. Ces corrections sont stockées
pour:
1. Analyse rétrospective des erreurs
2. Réentraînement ciblé du modèle
3. Amélioration continue des seuils

Workflow:
1. Le modèle fait une prédiction (cellules, TILs, mitoses)
2. Le pathologiste examine et signale les erreurs
3. Le feedback est stocké avec métadonnées
4. Agrégation périodique pour retraining
"""

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class FeedbackType(Enum):
    """Types de correction possibles."""
    # Classification cellulaire
    CELL_TYPE_WRONG = "cell_type_wrong"       # Mauvais type prédit
    CELL_MISSED = "cell_missed"               # Cellule non détectée (FN)
    CELL_FALSE_POSITIVE = "cell_fp"           # Fausse détection (FP)

    # Index mitotique
    MITOSIS_FALSE_POSITIVE = "mitosis_fp"     # Fausse mitose
    MITOSIS_MISSED = "mitosis_fn"             # Mitose manquée

    # TILs status
    TILS_STATUS_WRONG = "tils_wrong"          # Mauvais hot/cold

    # Organe
    ORGAN_WRONG = "organ_wrong"               # Mauvais organe détecté

    # Qualité générale
    OOD_MISSED = "ood_missed"                 # Devait être OOD
    FALSE_ALARM = "false_alarm"               # Alerte non justifiée

    # Commentaire libre
    OTHER = "other"


@dataclass
class FeedbackEntry:
    """Une entrée de feedback expert."""

    # Identifiant unique
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Contexte de l'image
    image_hash: str = ""                      # Hash de l'image (reproductibilité)
    patch_coordinates: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    organ_detected: str = ""
    organ_corrected: Optional[str] = None

    # Type de feedback
    feedback_type: FeedbackType = FeedbackType.OTHER

    # Détails de la correction
    nucleus_id: Optional[int] = None          # ID du noyau concerné
    nucleus_location: Optional[Tuple[int, int]] = None  # (y, x)
    predicted_class: Optional[str] = None     # Prédiction du modèle
    corrected_class: Optional[str] = None     # Correction de l'expert

    # Métadonnées modèle
    model_confidence: float = 0.0             # Confiance du modèle
    model_version: str = "optimus-gate-v1"

    # Commentaire expert
    expert_comment: str = ""
    expert_id: str = "anonymous"              # Optionnel, pour audit

    # Severity (pour priorisation du retraining)
    severity: str = "medium"                  # low, medium, high, critical

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire sérialisable."""
        d = asdict(self)
        d["feedback_type"] = self.feedback_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackEntry":
        """Crée une instance depuis un dictionnaire."""
        data["feedback_type"] = FeedbackType(data["feedback_type"])
        if data.get("patch_coordinates"):
            data["patch_coordinates"] = tuple(data["patch_coordinates"])
        if data.get("nucleus_location"):
            data["nucleus_location"] = tuple(data["nucleus_location"])
        return cls(**data)


class FeedbackCollector:
    """
    Collecteur de feedback pour Active Learning.

    Stocke les corrections des pathologistes dans un fichier JSON
    local pour analyse ultérieure et retraining.
    """

    def __init__(
        self,
        storage_path: str = "data/feedback",
        max_entries_per_file: int = 1000,
    ):
        """
        Args:
            storage_path: Dossier de stockage des feedbacks
            max_entries_per_file: Nombre max d'entrées par fichier
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries_per_file

        # Cache en mémoire pour la session courante
        self._session_entries: List[FeedbackEntry] = []
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_feedback(
        self,
        feedback_type: FeedbackType,
        image_hash: str = "",
        nucleus_id: Optional[int] = None,
        nucleus_location: Optional[Tuple[int, int]] = None,
        predicted_class: Optional[str] = None,
        corrected_class: Optional[str] = None,
        model_confidence: float = 0.0,
        organ_detected: str = "",
        organ_corrected: Optional[str] = None,
        expert_comment: str = "",
        expert_id: str = "anonymous",
        severity: str = "medium",
    ) -> FeedbackEntry:
        """
        Enregistre un feedback expert.

        Returns:
            L'entrée de feedback créée
        """
        entry = FeedbackEntry(
            image_hash=image_hash,
            feedback_type=feedback_type,
            nucleus_id=nucleus_id,
            nucleus_location=nucleus_location,
            predicted_class=predicted_class,
            corrected_class=corrected_class,
            model_confidence=model_confidence,
            organ_detected=organ_detected,
            organ_corrected=organ_corrected,
            expert_comment=expert_comment,
            expert_id=expert_id,
            severity=severity,
        )

        self._session_entries.append(entry)

        # Auto-save si le cache est plein
        if len(self._session_entries) >= self.max_entries:
            self.save_session()

        return entry

    def add_cell_type_correction(
        self,
        nucleus_id: int,
        nucleus_location: Tuple[int, int],
        predicted_class: str,
        corrected_class: str,
        model_confidence: float = 0.0,
        image_hash: str = "",
        expert_comment: str = "",
    ) -> FeedbackEntry:
        """Raccourci pour corriger un type cellulaire."""
        return self.add_feedback(
            feedback_type=FeedbackType.CELL_TYPE_WRONG,
            nucleus_id=nucleus_id,
            nucleus_location=nucleus_location,
            predicted_class=predicted_class,
            corrected_class=corrected_class,
            model_confidence=model_confidence,
            image_hash=image_hash,
            expert_comment=expert_comment,
            severity="high",
        )

    def add_mitosis_false_positive(
        self,
        nucleus_id: int,
        nucleus_location: Tuple[int, int],
        actual_type: str = "Not mitosis",
        image_hash: str = "",
        expert_comment: str = "",
    ) -> FeedbackEntry:
        """Signale une fausse mitose."""
        return self.add_feedback(
            feedback_type=FeedbackType.MITOSIS_FALSE_POSITIVE,
            nucleus_id=nucleus_id,
            nucleus_location=nucleus_location,
            predicted_class="Mitosis",
            corrected_class=actual_type,
            image_hash=image_hash,
            expert_comment=expert_comment,
            severity="high",  # Les fausses mitoses sont critiques
        )

    def add_mitosis_missed(
        self,
        nucleus_location: Tuple[int, int],
        image_hash: str = "",
        expert_comment: str = "",
    ) -> FeedbackEntry:
        """Signale une mitose manquée."""
        return self.add_feedback(
            feedback_type=FeedbackType.MITOSIS_MISSED,
            nucleus_location=nucleus_location,
            predicted_class="Not detected",
            corrected_class="Mitosis",
            image_hash=image_hash,
            expert_comment=expert_comment,
            severity="critical",  # Une mitose manquée est très grave
        )

    def add_tils_status_correction(
        self,
        predicted_status: str,
        corrected_status: str,
        image_hash: str = "",
        expert_comment: str = "",
    ) -> FeedbackEntry:
        """Corrige le statut TILs hot/cold."""
        return self.add_feedback(
            feedback_type=FeedbackType.TILS_STATUS_WRONG,
            predicted_class=predicted_status,
            corrected_class=corrected_status,
            image_hash=image_hash,
            expert_comment=expert_comment,
            severity="medium",
        )

    def add_organ_correction(
        self,
        predicted_organ: str,
        corrected_organ: str,
        model_confidence: float = 0.0,
        image_hash: str = "",
        expert_comment: str = "",
    ) -> FeedbackEntry:
        """Corrige l'organe détecté."""
        return self.add_feedback(
            feedback_type=FeedbackType.ORGAN_WRONG,
            organ_detected=predicted_organ,
            organ_corrected=corrected_organ,
            model_confidence=model_confidence,
            image_hash=image_hash,
            expert_comment=expert_comment,
            severity="high",  # Mauvais organe = mauvais décodeur
        )

    def save_session(self) -> str:
        """
        Sauvegarde les feedbacks de la session courante.

        Returns:
            Chemin du fichier sauvegardé
        """
        if not self._session_entries:
            return ""

        filename = f"feedback_{self._session_id}.json"
        filepath = self.storage_path / filename

        # Charger les données existantes si le fichier existe
        existing = []
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)

        # Ajouter les nouvelles entrées
        existing.extend([e.to_dict() for e in self._session_entries])

        # Sauvegarder
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        # Vider le cache
        self._session_entries = []

        return str(filepath)

    def load_all_feedback(self) -> List[FeedbackEntry]:
        """Charge tous les feedbacks stockés."""
        all_entries = []

        for filepath in self.storage_path.glob("feedback_*.json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_entries.extend([FeedbackEntry.from_dict(d) for d in data])

        return all_entries

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calcule des statistiques sur les feedbacks collectés.

        Utile pour identifier les patterns d'erreur et
        prioriser le retraining.
        """
        entries = self.load_all_feedback()

        if not entries:
            return {"total": 0, "message": "Aucun feedback collecté"}

        stats = {
            "total": len(entries),
            "by_type": {},
            "by_severity": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "by_organ": {},
            "mitosis_fp_rate": 0.0,
            "common_corrections": [],
        }

        # Par type
        for entry in entries:
            t = entry.feedback_type.value
            stats["by_type"][t] = stats["by_type"].get(t, 0) + 1
            stats["by_severity"][entry.severity] += 1

            if entry.organ_detected:
                org = entry.organ_detected
                stats["by_organ"][org] = stats["by_organ"].get(org, 0) + 1

        # Corrections les plus fréquentes
        corrections = {}
        for entry in entries:
            if entry.predicted_class and entry.corrected_class:
                key = f"{entry.predicted_class} -> {entry.corrected_class}"
                corrections[key] = corrections.get(key, 0) + 1

        stats["common_corrections"] = sorted(
            corrections.items(), key=lambda x: -x[1]
        )[:10]

        # Taux de faux positifs mitotiques
        mitosis_fp = stats["by_type"].get("mitosis_fp", 0)
        mitosis_fn = stats["by_type"].get("mitosis_fn", 0)
        if mitosis_fp + mitosis_fn > 0:
            stats["mitosis_fp_rate"] = mitosis_fp / (mitosis_fp + mitosis_fn)

        return stats

    def export_for_retraining(
        self,
        output_path: str,
        min_severity: str = "medium",
    ) -> str:
        """
        Exporte les feedbacks au format adapté pour le retraining.

        Args:
            output_path: Chemin du fichier de sortie
            min_severity: Niveau minimum de sévérité à inclure

        Returns:
            Chemin du fichier exporté
        """
        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_level = severity_order.get(min_severity, 1)

        entries = self.load_all_feedback()
        filtered = [
            e for e in entries
            if severity_order.get(e.severity, 1) >= min_level
        ]

        # Format pour retraining
        export_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "total_entries": len(filtered),
                "min_severity": min_severity,
            },
            "corrections": [e.to_dict() for e in filtered],
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return str(output_file)

    def get_session_summary(self) -> str:
        """Résumé de la session courante."""
        if not self._session_entries:
            return "Aucun feedback dans cette session."

        lines = [
            f"Session {self._session_id}",
            f"Feedbacks: {len(self._session_entries)}",
            "",
        ]

        by_type = {}
        for e in self._session_entries:
            t = e.feedback_type.value
            by_type[t] = by_type.get(t, 0) + 1

        for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
            lines.append(f"  - {t}: {count}")

        return "\n".join(lines)


# Singleton pour accès global
_collector: Optional[FeedbackCollector] = None


def get_feedback_collector(storage_path: str = "data/feedback") -> FeedbackCollector:
    """Retourne le collecteur de feedback (singleton)."""
    global _collector
    if _collector is None:
        _collector = FeedbackCollector(storage_path)
    return _collector


# Test
if __name__ == "__main__":
    print("Test FeedbackCollector...")

    collector = FeedbackCollector(storage_path="/tmp/test_feedback")

    # Simuler des corrections
    collector.add_cell_type_correction(
        nucleus_id=42,
        nucleus_location=(100, 150),
        predicted_class="Neoplastic",
        corrected_class="Inflammatory",
        model_confidence=0.85,
        expert_comment="Lymphocyte clairement identifiable",
    )

    collector.add_mitosis_false_positive(
        nucleus_id=17,
        nucleus_location=(200, 180),
        actual_type="Fibroblast",
        expert_comment="Cellule allongée mais pas de chromatine condensée",
    )

    collector.add_tils_status_correction(
        predicted_status="froid",
        corrected_status="chaud",
        expert_comment="TILs visibles au centre de la tumeur",
    )

    # Sauvegarder
    path = collector.save_session()
    print(f"Sauvegardé: {path}")

    # Statistiques
    stats = collector.get_statistics()
    print(f"\nStatistiques:")
    print(f"  Total: {stats['total']}")
    print(f"  Par type: {stats['by_type']}")
    print(f"  Par sévérité: {stats['by_severity']}")

    print("\n Active Learning module ready!")
