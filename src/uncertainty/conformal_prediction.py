#!/usr/bin/env python3
"""
Conformal Prediction pour CellViT-Optimus.

Fournit des ensembles de prédiction avec garanties de couverture.
Référence: Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction", 2022.

Usage:
    # Calibrer sur données de validation
    conformal = ConformalPredictor()
    conformal.calibrate(val_probs, val_labels, alpha=0.1)  # 90% coverage

    # Prédire avec ensemble
    pred_set = conformal.predict_set(test_probs)  # Set de classes possibles
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Set, Optional, Tuple
from enum import Enum


class ConformalMethod(Enum):
    """Méthodes de conformal prediction."""
    LAC = "lac"          # Least Ambiguous set-valued Classifier
    APS = "aps"          # Adaptive Prediction Sets
    RAPS = "raps"        # Regularized APS


@dataclass
class ConformalResult:
    """Résultat de la prédiction conforme."""
    prediction_set: Set[int]      # Ensemble de classes possibles
    set_size: int                  # Taille de l'ensemble
    confidence: float              # Confiance de la prédiction principale
    coverage_guaranteed: float     # Couverture garantie (1 - alpha)
    is_singleton: bool             # True si une seule classe
    is_empty: bool                 # True si ensemble vide (rare)

    @property
    def is_uncertain(self) -> bool:
        """True si plus d'une classe dans l'ensemble."""
        return self.set_size > 1

    def get_classes_str(self, class_names: Optional[List[str]] = None) -> str:
        """Retourne les classes sous forme de string."""
        if class_names:
            return ", ".join(class_names[i] for i in sorted(self.prediction_set))
        return ", ".join(str(i) for i in sorted(self.prediction_set))


class ConformalPredictor:
    """
    Prédicteur conforme pour classification.

    Garantit que la vraie classe est dans l'ensemble prédit avec
    probabilité >= 1 - alpha (ex: 90% pour alpha=0.1).
    """

    def __init__(
        self,
        method: ConformalMethod = ConformalMethod.APS,
        alpha: float = 0.1,
        regularization: float = 0.0,  # Pour RAPS
    ):
        """
        Args:
            method: Méthode de conformal prediction
            alpha: Niveau d'erreur (1-alpha = couverture)
            regularization: Pénalité pour grands ensembles (RAPS)
        """
        self.method = method
        self.alpha = alpha
        self.regularization = regularization

        # Calibration
        self.threshold = None
        self.calibrated = False
        self.n_calibration = 0

    def _compute_scores(
        self,
        probs: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calcule les scores de non-conformité.

        Pour LAC: 1 - prob[true_class]
        Pour APS: Somme cumulative jusqu'à la vraie classe
        """
        if self.method == ConformalMethod.LAC:
            if labels is None:
                # Pour prédiction: 1 - max_prob
                return 1 - probs.max(axis=-1)
            else:
                # Pour calibration: 1 - prob[label]
                return 1 - probs[np.arange(len(labels)), labels]

        elif self.method in [ConformalMethod.APS, ConformalMethod.RAPS]:
            # Trier par probabilité décroissante
            sorted_indices = np.argsort(-probs, axis=-1)
            sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)

            # Somme cumulative
            cumsum = np.cumsum(sorted_probs, axis=-1)

            if labels is None:
                # Pour prédiction: on retourne les cumsums
                return cumsum, sorted_indices
            else:
                # Pour calibration: score = cumsum jusqu'à la vraie classe
                # Trouver la position de la vraie classe
                n_samples, n_classes = probs.shape
                true_positions = np.zeros(n_samples, dtype=int)

                for i in range(n_samples):
                    true_positions[i] = np.where(sorted_indices[i] == labels[i])[0][0]

                scores = cumsum[np.arange(n_samples), true_positions]

                # RAPS: ajouter pénalité
                if self.method == ConformalMethod.RAPS:
                    scores += self.regularization * (true_positions + 1)

                return scores

        raise ValueError(f"Méthode inconnue: {self.method}")

    def calibrate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        alpha: Optional[float] = None
    ) -> float:
        """
        Calibre le seuil sur des données de validation.

        Args:
            probs: Probabilités de validation (N, C)
            labels: Labels ground truth (N,)
            alpha: Niveau d'erreur (optionnel, sinon utilise self.alpha)

        Returns:
            Seuil calibré
        """
        if alpha is not None:
            self.alpha = alpha

        # Calculer les scores
        scores = self._compute_scores(probs, labels)

        # Calculer le quantile pour garantir la couverture
        n = len(scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        q = min(q, 1.0)

        self.threshold = np.quantile(scores, q)
        self.calibrated = True
        self.n_calibration = n

        return self.threshold

    def predict_set(self, probs: np.ndarray) -> List[ConformalResult]:
        """
        Prédit des ensembles de classes.

        Args:
            probs: Probabilités (N, C) ou (C,)

        Returns:
            Liste de ConformalResult
        """
        if not self.calibrated:
            raise RuntimeError("Appeler calibrate() d'abord")

        # Gérer le cas 1D
        single = probs.ndim == 1
        if single:
            probs = probs.reshape(1, -1)

        n_samples, n_classes = probs.shape
        results = []

        if self.method == ConformalMethod.LAC:
            for i in range(n_samples):
                # Inclure toutes les classes avec prob > 1 - threshold
                pred_set = set(
                    j for j in range(n_classes)
                    if probs[i, j] > 1 - self.threshold
                )

                # Au minimum la classe la plus probable
                if len(pred_set) == 0:
                    pred_set = {probs[i].argmax()}

                results.append(ConformalResult(
                    prediction_set=pred_set,
                    set_size=len(pred_set),
                    confidence=probs[i].max(),
                    coverage_guaranteed=1 - self.alpha,
                    is_singleton=len(pred_set) == 1,
                    is_empty=False,
                ))

        elif self.method in [ConformalMethod.APS, ConformalMethod.RAPS]:
            cumsum, sorted_indices = self._compute_scores(probs)

            for i in range(n_samples):
                # Inclure les classes jusqu'à ce que cumsum > threshold
                pred_set = set()
                for k in range(n_classes):
                    pred_set.add(int(sorted_indices[i, k]))

                    score = cumsum[i, k]
                    if self.method == ConformalMethod.RAPS:
                        score += self.regularization * (k + 1)

                    if score >= self.threshold:
                        break

                results.append(ConformalResult(
                    prediction_set=pred_set,
                    set_size=len(pred_set),
                    confidence=probs[i].max(),
                    coverage_guaranteed=1 - self.alpha,
                    is_singleton=len(pred_set) == 1,
                    is_empty=len(pred_set) == 0,
                ))

        if single:
            return results[0]
        return results

    def evaluate_coverage(
        self,
        probs: np.ndarray,
        labels: np.ndarray
    ) -> dict:
        """
        Évalue la couverture et la taille des ensembles.

        Returns:
            Dict avec métriques de performance
        """
        results = self.predict_set(probs)

        # Couverture empirique
        covered = sum(
            1 for r, l in zip(results, labels)
            if l in r.prediction_set
        )
        coverage = covered / len(labels)

        # Taille moyenne des ensembles
        avg_size = np.mean([r.set_size for r in results])

        # Proportion de singletons
        singletons = sum(r.is_singleton for r in results) / len(results)

        return {
            'coverage': coverage,
            'target_coverage': 1 - self.alpha,
            'avg_set_size': avg_size,
            'singleton_rate': singletons,
            'n_samples': len(labels),
        }


class PixelwiseConformalPredictor:
    """
    Prédicteur conforme pour segmentation pixel par pixel.

    Adapte la prédiction conforme au cas de segmentation où chaque
    pixel a sa propre distribution de probabilités.
    """

    def __init__(
        self,
        method: ConformalMethod = ConformalMethod.APS,
        alpha: float = 0.1,
    ):
        self.predictor = ConformalPredictor(method=method, alpha=alpha)

    def calibrate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        sample_pixels: int = 10000
    ) -> float:
        """
        Calibre sur un échantillon de pixels.

        Args:
            probs: Probabilités (N, H, W, C) ou (H, W, C)
            labels: Labels (N, H, W) ou (H, W)
            sample_pixels: Nombre de pixels à échantillonner
        """
        # Aplatir pour échantillonnage
        if probs.ndim == 3:
            probs = probs.reshape(-1, probs.shape[-1])
            labels = labels.reshape(-1)
        else:
            probs = probs.reshape(-1, probs.shape[-1])
            labels = labels.reshape(-1)

        # Échantillonner
        if len(probs) > sample_pixels:
            indices = np.random.choice(len(probs), sample_pixels, replace=False)
            probs = probs[indices]
            labels = labels[indices]

        return self.predictor.calibrate(probs, labels)

    def predict_set_map(self, probs: np.ndarray) -> np.ndarray:
        """
        Prédit une carte de taille d'ensemble.

        Args:
            probs: Probabilités (H, W, C)

        Returns:
            Carte (H, W) avec taille d'ensemble par pixel
        """
        H, W, C = probs.shape
        flat_probs = probs.reshape(-1, C)

        results = self.predictor.predict_set(flat_probs)
        sizes = np.array([r.set_size for r in results])

        return sizes.reshape(H, W)

    def get_uncertain_mask(self, probs: np.ndarray) -> np.ndarray:
        """
        Retourne un masque des pixels incertains (set_size > 1).
        """
        size_map = self.predict_set_map(probs)
        return size_map > 1


# Tests
if __name__ == "__main__":
    print("Test ConformalPredictor...")
    print("=" * 50)

    np.random.seed(42)

    # Simuler des données
    n_cal = 500
    n_test = 100
    n_classes = 5

    # Probabilités simulées (softmax de scores aléatoires)
    from scipy.special import softmax

    cal_logits = np.random.randn(n_cal, n_classes)
    cal_probs = softmax(cal_logits, axis=-1)
    cal_labels = np.random.randint(0, n_classes, n_cal)

    test_logits = np.random.randn(n_test, n_classes)
    test_probs = softmax(test_logits, axis=-1)
    test_labels = np.random.randint(0, n_classes, n_test)

    # Tester les différentes méthodes
    for method in ConformalMethod:
        print(f"\n{method.value.upper()}:")

        cp = ConformalPredictor(method=method, alpha=0.1)
        threshold = cp.calibrate(cal_probs, cal_labels)
        print(f"  Threshold: {threshold:.4f}")

        metrics = cp.evaluate_coverage(test_probs, test_labels)
        print(f"  Coverage: {metrics['coverage']:.3f} (target: {metrics['target_coverage']:.3f})")
        print(f"  Avg set size: {metrics['avg_set_size']:.2f}")
        print(f"  Singleton rate: {metrics['singleton_rate']:.3f}")

    # Test sur une seule prédiction
    print("\n" + "=" * 50)
    print("Test prédiction unique:")

    cp = ConformalPredictor(method=ConformalMethod.APS, alpha=0.1)
    cp.calibrate(cal_probs, cal_labels)

    result = cp.predict_set(test_probs[0])
    class_names = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]

    print(f"  Prediction set: {result.get_classes_str(class_names)}")
    print(f"  Set size: {result.set_size}")
    print(f"  Is uncertain: {result.is_uncertain}")
    print(f"  Confidence: {result.confidence:.3f}")

    print("\n✅ Tests passés!")
