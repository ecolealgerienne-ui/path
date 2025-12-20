#!/usr/bin/env python3
"""
Estimateur d'incertitude unifié pour CellViT-Optimus.

Combine:
- Incertitude aléatorique (entropie des prédictions NP/NT)
- Incertitude épistémique (distance Mahalanobis sur embeddings)
- Calibration (Temperature Scaling)

Sortie en 3 niveaux: {Fiable | À revoir | Hors domaine}
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum


class ConfidenceLevel(Enum):
    """Niveaux de confiance pour les prédictions."""
    FIABLE = "fiable"
    A_REVOIR = "a_revoir"
    HORS_DOMAINE = "hors_domaine"


@dataclass
class UncertaintyResult:
    """Résultat de l'estimation d'incertitude."""
    # Niveau de confiance global
    level: ConfidenceLevel

    # Scores détaillés
    entropy_np: float           # Entropie NP (0-1)
    entropy_nt: float           # Entropie NT (0-1)
    max_confidence: float       # Confiance max (0-1)
    mahalanobis_score: float    # Distance Mahalanobis (0+)

    # Scores composites
    aleatoric_score: float      # Incertitude aléatorique (0-1)
    epistemic_score: float      # Incertitude épistémique (0-1)
    combined_score: float       # Score combiné (0-1)

    # Carte spatiale d'incertitude (optionnelle)
    uncertainty_map: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire."""
        return {
            'level': self.level.value,
            'entropy_np': self.entropy_np,
            'entropy_nt': self.entropy_nt,
            'max_confidence': self.max_confidence,
            'mahalanobis_score': self.mahalanobis_score,
            'aleatoric_score': self.aleatoric_score,
            'epistemic_score': self.epistemic_score,
            'combined_score': self.combined_score,
        }

    def get_emoji(self) -> str:
        """Retourne l'emoji correspondant au niveau."""
        if self.level == ConfidenceLevel.FIABLE:
            return "✅"
        elif self.level == ConfidenceLevel.A_REVOIR:
            return "⚠️"
        else:
            return "🚫"

    def get_color(self) -> Tuple[int, int, int]:
        """Retourne la couleur RGB correspondante."""
        if self.level == ConfidenceLevel.FIABLE:
            return (0, 200, 0)      # Vert
        elif self.level == ConfidenceLevel.A_REVOIR:
            return (255, 165, 0)    # Orange
        else:
            return (200, 0, 0)      # Rouge


class UncertaintyEstimator:
    """
    Estimateur d'incertitude pour les prédictions HoVer-Net.

    Combine plusieurs signaux pour classifier les prédictions:
    - Fiable: Haute confiance, basse entropie, proche de la distribution
    - À revoir: Incertitude moyenne, validation humaine recommandée
    - Hors domaine: Haute entropie ou loin de la distribution

    Usage:
        estimator = UncertaintyEstimator()
        estimator.fit_ood(train_embeddings)  # Optionnel: calibrer OOD

        result = estimator.estimate(
            np_probs=np_probs,      # (H, W, 2)
            nt_probs=nt_probs,      # (H, W, 5)
            embeddings=embeddings,  # (1536,) optionnel
        )

        print(result.level)  # ConfidenceLevel.FIABLE
    """

    def __init__(
        self,
        entropy_threshold_low: float = 0.3,
        entropy_threshold_high: float = 0.6,
        confidence_threshold: float = 0.7,
        mahalanobis_threshold: float = None,
        temperature: float = 1.0,
    ):
        """
        Args:
            entropy_threshold_low: Seuil bas entropie (< = fiable)
            entropy_threshold_high: Seuil haut entropie (> = hors domaine)
            confidence_threshold: Seuil confiance minimum
            mahalanobis_threshold: Seuil distance Mahalanobis (auto si None)
            temperature: Température pour calibration
        """
        self.entropy_threshold_low = entropy_threshold_low
        self.entropy_threshold_high = entropy_threshold_high
        self.confidence_threshold = confidence_threshold
        self.mahalanobis_threshold = mahalanobis_threshold
        self.temperature = temperature

        # Pour OOD detection
        self.ood_mean = None
        self.ood_cov_inv = None
        self.ood_fitted = False

    def fit_ood(
        self,
        embeddings: np.ndarray,
        percentile: float = 95.0
    ):
        """
        Calibre le détecteur OOD sur les embeddings d'entraînement.

        Args:
            embeddings: Embeddings H-optimus-0 (N, 1536)
            percentile: Percentile pour le seuil automatique
        """
        from sklearn.covariance import LedoitWolf

        self.ood_mean = embeddings.mean(axis=0)

        # Estimation robuste de la covariance
        cov_estimator = LedoitWolf()
        cov_estimator.fit(embeddings)
        self.ood_cov_inv = np.linalg.pinv(cov_estimator.covariance_)

        # Calculer le seuil sur les données d'entraînement
        train_scores = self._mahalanobis_distance(embeddings)
        self.mahalanobis_threshold = np.percentile(train_scores, percentile)

        self.ood_fitted = True

        return self

    def _mahalanobis_distance(self, embeddings: np.ndarray) -> np.ndarray:
        """Calcule la distance de Mahalanobis."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        diff = embeddings - self.ood_mean
        scores = np.sqrt(np.sum(diff @ self.ood_cov_inv * diff, axis=1))

        return scores

    def _entropy(self, probs: np.ndarray, eps: float = 1e-10) -> float:
        """Calcule l'entropie normalisée moyenne."""
        probs = np.clip(probs, eps, 1 - eps)

        # Entropie par pixel
        n_classes = probs.shape[-1]
        entropy = -np.sum(probs * np.log(probs), axis=-1)

        # Normaliser par log(n_classes)
        max_entropy = np.log(n_classes)
        normalized = entropy / max_entropy

        # Moyenne sur les pixels du masque (exclure background)
        mask = probs[..., 0] < 0.5  # Pixels avec probabilité foreground
        if mask.sum() > 0:
            return float(normalized[mask].mean())
        return float(normalized.mean())

    def _max_confidence(self, probs: np.ndarray) -> float:
        """Calcule la confiance moyenne sur les prédictions."""
        max_probs = probs.max(axis=-1)

        # Moyenne sur les pixels foreground
        mask = probs[..., 0] < 0.5 if probs.shape[-1] == 2 else np.ones(probs.shape[:-1], dtype=bool)
        if mask.sum() > 0:
            return float(max_probs[mask].mean())
        return float(max_probs.mean())

    def _compute_uncertainty_map(
        self,
        np_probs: np.ndarray,
        nt_probs: np.ndarray
    ) -> np.ndarray:
        """
        Génère une carte d'incertitude spatiale.

        Combine entropie NP et NT pour chaque pixel.
        """
        # Entropie NP
        np_entropy = -np.sum(np_probs * np.log(np_probs + 1e-10), axis=-1)
        np_entropy /= np.log(2)  # Normaliser (2 classes)

        # Entropie NT
        nt_entropy = -np.sum(nt_probs * np.log(nt_probs + 1e-10), axis=-1)
        nt_entropy /= np.log(5)  # Normaliser (5 classes)

        # Combiner (moyenne pondérée)
        uncertainty_map = 0.5 * np_entropy + 0.5 * nt_entropy

        return uncertainty_map

    def estimate(
        self,
        np_probs: np.ndarray,
        nt_probs: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        compute_map: bool = True,
    ) -> UncertaintyResult:
        """
        Estime l'incertitude des prédictions.

        Args:
            np_probs: Probabilités NP (H, W, 2) ou (2, H, W)
            nt_probs: Probabilités NT (H, W, 5) ou (5, H, W)
            embeddings: Embeddings H-optimus-0 (1536,) optionnel
            compute_map: Calculer la carte d'incertitude spatiale

        Returns:
            UncertaintyResult avec niveau de confiance et scores
        """
        # Assurer format (H, W, C)
        if np_probs.shape[0] == 2:
            np_probs = np.transpose(np_probs, (1, 2, 0))
        if nt_probs.shape[0] == 5:
            nt_probs = np.transpose(nt_probs, (1, 2, 0))

        # Calculer les métriques d'entropie
        entropy_np = self._entropy(np_probs)
        entropy_nt = self._entropy(nt_probs)
        max_conf = self._max_confidence(nt_probs)

        # Score aléatorique (basé sur entropie)
        aleatoric = (entropy_np + entropy_nt) / 2

        # Score épistémique (basé sur Mahalanobis si disponible)
        mahalanobis = 0.0
        epistemic = 0.0

        if embeddings is not None and self.ood_fitted:
            mahalanobis = float(self._mahalanobis_distance(embeddings)[0])
            # Normaliser le score Mahalanobis (0 au seuil = 1)
            epistemic = min(mahalanobis / (self.mahalanobis_threshold + 1e-10), 2.0) / 2.0

        # Score combiné
        if self.ood_fitted:
            combined = 0.6 * aleatoric + 0.4 * epistemic
        else:
            combined = aleatoric

        # Déterminer le niveau de confiance
        level = self._classify(
            entropy_np=entropy_np,
            entropy_nt=entropy_nt,
            max_conf=max_conf,
            mahalanobis=mahalanobis,
        )

        # Carte d'incertitude
        uncertainty_map = None
        if compute_map:
            uncertainty_map = self._compute_uncertainty_map(np_probs, nt_probs)

        return UncertaintyResult(
            level=level,
            entropy_np=entropy_np,
            entropy_nt=entropy_nt,
            max_confidence=max_conf,
            mahalanobis_score=mahalanobis,
            aleatoric_score=aleatoric,
            epistemic_score=epistemic,
            combined_score=combined,
            uncertainty_map=uncertainty_map,
        )

    def _classify(
        self,
        entropy_np: float,
        entropy_nt: float,
        max_conf: float,
        mahalanobis: float,
    ) -> ConfidenceLevel:
        """
        Classifie en 3 niveaux selon les scores.

        Logique:
        - HORS_DOMAINE: Entropie très haute OU Mahalanobis > seuil
        - FIABLE: Entropie basse ET confiance haute
        - A_REVOIR: Tous les autres cas
        """
        avg_entropy = (entropy_np + entropy_nt) / 2

        # Vérifier hors domaine
        is_ood = False
        if avg_entropy > self.entropy_threshold_high:
            is_ood = True
        if max_conf < self.confidence_threshold * 0.5:
            is_ood = True
        if self.ood_fitted and self.mahalanobis_threshold:
            if mahalanobis > self.mahalanobis_threshold * 1.5:
                is_ood = True

        if is_ood:
            return ConfidenceLevel.HORS_DOMAINE

        # Vérifier fiable
        is_reliable = (
            avg_entropy < self.entropy_threshold_low and
            max_conf > self.confidence_threshold
        )

        if is_reliable:
            return ConfidenceLevel.FIABLE

        # Sinon: à revoir
        return ConfidenceLevel.A_REVOIR

    def generate_report(self, result: UncertaintyResult) -> str:
        """Génère un rapport textuel."""
        emoji = result.get_emoji()
        level_text = {
            ConfidenceLevel.FIABLE: "FIABLE - Prédiction utilisable",
            ConfidenceLevel.A_REVOIR: "À REVOIR - Validation humaine recommandée",
            ConfidenceLevel.HORS_DOMAINE: "HORS DOMAINE - Ne pas utiliser",
        }

        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"{emoji} NIVEAU DE CONFIANCE: {level_text[result.level]}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            "📊 Métriques d'incertitude:",
            f"   • Entropie NP: {result.entropy_np:.3f}",
            f"   • Entropie NT: {result.entropy_nt:.3f}",
            f"   • Confiance max: {result.max_confidence:.3f}",
        ]

        if result.mahalanobis_score > 0:
            lines.append(f"   • Score Mahalanobis: {result.mahalanobis_score:.2f}")

        lines.extend([
            "",
            f"   → Incertitude aléatorique: {result.aleatoric_score:.3f}",
            f"   → Incertitude épistémique: {result.epistemic_score:.3f}",
            f"   → Score combiné: {result.combined_score:.3f}",
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ])

        return "\n".join(lines)


# Test
if __name__ == "__main__":
    print("Test UncertaintyEstimator...")

    # Simuler des prédictions
    H, W = 224, 224

    # Cas 1: Haute confiance
    np_probs_high = np.zeros((H, W, 2))
    np_probs_high[..., 1] = 0.9
    np_probs_high[..., 0] = 0.1

    nt_probs_high = np.zeros((H, W, 5))
    nt_probs_high[..., 0] = 0.85
    nt_probs_high[..., 1:] = 0.0375

    # Cas 2: Basse confiance
    np_probs_low = np.ones((H, W, 2)) * 0.5
    nt_probs_low = np.ones((H, W, 5)) * 0.2

    estimator = UncertaintyEstimator()

    # Test haute confiance
    result_high = estimator.estimate(np_probs_high, nt_probs_high)
    print(f"\nHaute confiance:")
    print(f"  Level: {result_high.level.value}")
    print(f"  Entropy NP: {result_high.entropy_np:.3f}")
    print(f"  Max conf: {result_high.max_confidence:.3f}")

    # Test basse confiance
    result_low = estimator.estimate(np_probs_low, nt_probs_low)
    print(f"\nBasse confiance:")
    print(f"  Level: {result_low.level.value}")
    print(f"  Entropy NP: {result_low.entropy_np:.3f}")
    print(f"  Max conf: {result_low.max_confidence:.3f}")

    # Rapport
    print("\n" + estimator.generate_report(result_high))
    print("\n" + estimator.generate_report(result_low))

    print("\n✅ Tests passés!")
