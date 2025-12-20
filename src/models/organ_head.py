#!/usr/bin/env python3
"""
Organ Classification Head pour CellViT-Optimus.

Utilise le CLS token de H-optimus-0 pour:
1. Classifier l'organe source (19 classes PanNuke)
2. Détecter les images hors distribution (OOD)

Architecture "Optimus-Gate":
    H-optimus-0 → CLS token (1536) → LayerNorm → MLP → 19 organes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


# Mapping PanNuke organs (19 classes)
PANNUKE_ORGANS = [
    "Adrenal_gland",
    "Bile-duct",
    "Bladder",
    "Breast",
    "Cervix",
    "Colon",
    "Esophagus",
    "HeadNeck",
    "Kidney",
    "Liver",
    "Lung",
    "Ovarian",
    "Pancreatic",
    "Prostate",
    "Skin",
    "Stomach",
    "Testis",
    "Thyroid",
    "Uterus",
]


@dataclass
class OrganPrediction:
    """Résultat de la prédiction d'organe."""
    organ_idx: int              # Index de l'organe prédit
    organ_name: str             # Nom de l'organe
    confidence: float           # Confiance (0-1)
    entropy: float              # Entropie normalisée (0-1)
    probabilities: np.ndarray   # Probabilités pour chaque organe
    is_ood: bool                # True si détecté comme OOD
    ood_score: float            # Score OOD combiné

    def to_dict(self) -> Dict:
        return {
            'organ_idx': self.organ_idx,
            'organ_name': self.organ_name,
            'confidence': self.confidence,
            'entropy': self.entropy,
            'is_ood': self.is_ood,
            'ood_score': self.ood_score,
        }


class OrganHead(nn.Module):
    """
    Tête de classification d'organe.

    Utilise le CLS token pour prédire l'organe source.
    Architecture simple mais efficace car H-optimus-0 a déjà
    des représentations d'organes bien structurées.
    """

    def __init__(
        self,
        embed_dim: int = 1536,
        hidden_dim: int = 256,
        n_organs: int = 19,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Dimension des embeddings H-optimus-0
            hidden_dim: Dimension de la couche cachée
            n_organs: Nombre d'organes (19 pour PanNuke)
            dropout: Dropout pour régularisation
        """
        super().__init__()

        self.n_organs = n_organs
        self.organ_names = PANNUKE_ORGANS[:n_organs]

        # Architecture: LayerNorm → Linear → ReLU → Dropout → Linear
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_organs),
        )

        # Pour OOD detection (Mahalanobis)
        self.register_buffer('cls_mean', None)
        self.register_buffer('cls_cov_inv', None)
        self.ood_fitted = False
        self.mahalanobis_threshold = None

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            cls_token: CLS token de H-optimus-0 (B, 1536)

        Returns:
            logits: Logits pour chaque organe (B, n_organs)
        """
        return self.classifier(cls_token)

    def predict(
        self,
        cls_token: torch.Tensor,
        return_probs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prédit l'organe avec probabilités.

        Args:
            cls_token: CLS token (B, 1536) ou (1536,)
            return_probs: Retourner les probabilités softmax

        Returns:
            predictions: Indices des organes prédits (B,)
            probs_or_logits: Probabilités ou logits (B, n_organs)
        """
        # Handle single sample
        if cls_token.dim() == 1:
            cls_token = cls_token.unsqueeze(0)

        logits = self.forward(cls_token)

        if return_probs:
            probs = F.softmax(logits, dim=-1)
            predictions = probs.argmax(dim=-1)
            return predictions, probs
        else:
            predictions = logits.argmax(dim=-1)
            return predictions, logits

    def compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Calcule l'entropie normalisée des probabilités.

        Args:
            probs: Probabilités (B, n_organs)

        Returns:
            entropy: Entropie normalisée [0, 1] (B,)
        """
        eps = 1e-10
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        max_entropy = np.log(self.n_organs)
        return entropy / max_entropy

    def fit_ood(
        self,
        cls_tokens: torch.Tensor,
        percentile: float = 95.0
    ):
        """
        Calibre le détecteur OOD sur les CLS tokens d'entraînement.

        Args:
            cls_tokens: CLS tokens d'entraînement (N, 1536)
            percentile: Percentile pour le seuil automatique
        """
        with torch.no_grad():
            # Convertir en numpy
            tokens_np = cls_tokens.cpu().numpy()

            # Mean
            mean = tokens_np.mean(axis=0)
            self.cls_mean = torch.from_numpy(mean).to(cls_tokens.device)

            # Covariance inverse (régularisée avec Ledoit-Wolf si disponible)
            try:
                from sklearn.covariance import LedoitWolf
                cov_estimator = LedoitWolf()
                cov_estimator.fit(tokens_np)
                cov = cov_estimator.covariance_
            except ImportError:
                # Fallback: covariance empirique avec régularisation
                cov = np.cov(tokens_np, rowvar=False)
                # Régularisation de Tikhonov
                reg = 1e-5 * np.trace(cov) / cov.shape[0]
                cov += reg * np.eye(cov.shape[0])

            cov_inv = np.linalg.pinv(cov)
            self.cls_cov_inv = torch.from_numpy(cov_inv).float().to(cls_tokens.device)

            # Calculer le seuil
            distances = self._mahalanobis_distance(cls_tokens)
            self.mahalanobis_threshold = float(np.percentile(
                distances.cpu().numpy(), percentile
            ))

            self.ood_fitted = True

    def _mahalanobis_distance(self, cls_tokens: torch.Tensor) -> torch.Tensor:
        """Calcule la distance de Mahalanobis."""
        if cls_tokens.dim() == 1:
            cls_tokens = cls_tokens.unsqueeze(0)

        diff = cls_tokens - self.cls_mean
        # (B, D) @ (D, D) @ (D, B) -> diagonale = (B,)
        left = torch.mm(diff, self.cls_cov_inv)
        distances = torch.sqrt(torch.sum(left * diff, dim=1))

        return distances

    def compute_ood_score(
        self,
        cls_token: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        entropy_weight: float = 0.5,
        mahalanobis_weight: float = 0.5,
    ) -> Tuple[float, bool]:
        """
        Calcule le score OOD combiné.

        Args:
            cls_token: CLS token (1536,) ou (1, 1536)
            probs: Probabilités softmax (optionnel)
            entropy_weight: Poids de l'entropie
            mahalanobis_weight: Poids du Mahalanobis

        Returns:
            ood_score: Score OOD [0, 1]
            is_ood: True si OOD détecté
        """
        if cls_token.dim() == 1:
            cls_token = cls_token.unsqueeze(0)

        # Calculer probs si non fourni
        if probs is None:
            _, probs = self.predict(cls_token)

        # Entropie normalisée
        entropy = self.compute_entropy(probs).item()

        # Mahalanobis normalisé
        mahal_score = 0.0
        if self.ood_fitted:
            mahal_dist = self._mahalanobis_distance(cls_token).item()
            # Normaliser par le seuil (1.0 au seuil)
            mahal_score = min(mahal_dist / (self.mahalanobis_threshold + 1e-10), 2.0) / 2.0

        # Score combiné
        ood_score = entropy_weight * entropy + mahalanobis_weight * mahal_score

        # Détection OOD (seuils relaxés pour éviter faux positifs)
        is_ood = False
        if entropy > 0.8:  # Entropie très haute (était 0.7)
            is_ood = True
        if self.ood_fitted and mahal_score > 1.0:  # Au-delà du seuil
            is_ood = True
        if ood_score > 0.75:  # Score combiné élevé (était 0.6)
            is_ood = True

        return ood_score, is_ood

    def predict_with_ood(
        self,
        cls_token: torch.Tensor,
    ) -> OrganPrediction:
        """
        Prédit l'organe avec détection OOD.

        Args:
            cls_token: CLS token (1536,)

        Returns:
            OrganPrediction avec toutes les informations
        """
        if cls_token.dim() == 1:
            cls_token = cls_token.unsqueeze(0)

        # Prédiction
        pred_idx, probs = self.predict(cls_token)
        pred_idx = pred_idx.item()
        probs_np = probs.squeeze().detach().cpu().numpy()

        # Métriques
        confidence = float(probs.max().detach())
        entropy = self.compute_entropy(probs.detach()).item()

        # OOD
        ood_score, is_ood = self.compute_ood_score(cls_token, probs)

        return OrganPrediction(
            organ_idx=pred_idx,
            organ_name=self.organ_names[pred_idx],
            confidence=confidence,
            entropy=entropy,
            probabilities=probs_np,
            is_ood=is_ood,
            ood_score=ood_score,
        )


class OrganHeadLoss(nn.Module):
    """
    Loss pour la tête organe avec class weights.

    Gère le déséquilibre des classes dans PanNuke.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
    ):
        """
        Args:
            class_weights: Poids par classe (n_organs,)
            label_smoothing: Label smoothing pour régularisation
        """
        super().__init__()
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Logits (B, n_organs)
            targets: Labels (B,)
        """
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    @staticmethod
    def compute_class_weights(
        labels: np.ndarray,
        n_classes: int = 19
    ) -> torch.Tensor:
        """
        Calcule les poids de classe inversement proportionnels à la fréquence.

        Args:
            labels: Labels d'entraînement (N,)
            n_classes: Nombre de classes

        Returns:
            weights: Poids par classe (n_classes,)
        """
        counts = np.bincount(labels, minlength=n_classes)
        # Éviter division par zéro
        counts = np.maximum(counts, 1)

        # Poids inversement proportionnels
        weights = len(labels) / (n_classes * counts)

        # Normaliser pour que mean = 1
        weights = weights / weights.mean()

        return torch.from_numpy(weights).float()


# Test
if __name__ == "__main__":
    print("Test OrganHead...")
    print("=" * 50)

    # Créer le modèle
    model = OrganHead(embed_dim=1536, hidden_dim=256, n_organs=19)
    model.eval()

    # Simuler un CLS token
    cls_token = torch.randn(1, 1536)

    # Forward
    logits = model(cls_token)
    print(f"✓ Logits shape: {logits.shape}")

    # Predict
    pred, probs = model.predict(cls_token)
    print(f"✓ Predicted organ: {model.organ_names[pred.item()]}")
    print(f"✓ Confidence: {probs.max().item():.3f}")

    # Entropy
    entropy = model.compute_entropy(probs)
    print(f"✓ Entropy: {entropy.item():.3f}")

    # Fit OOD
    train_tokens = torch.randn(100, 1536)
    model.fit_ood(train_tokens)
    print(f"✓ OOD fitted, threshold: {model.mahalanobis_threshold:.2f}")

    # Full prediction with OOD
    result = model.predict_with_ood(cls_token.squeeze())
    print(f"\n📋 Résultat complet:")
    print(f"   Organe: {result.organ_name}")
    print(f"   Confiance: {result.confidence:.3f}")
    print(f"   Entropie: {result.entropy:.3f}")
    print(f"   Score OOD: {result.ood_score:.3f}")
    print(f"   Is OOD: {result.is_ood}")

    # Test loss
    print(f"\n✓ Test loss...")
    labels = torch.randint(0, 19, (100,))
    weights = OrganHeadLoss.compute_class_weights(labels.numpy())
    loss_fn = OrganHeadLoss(class_weights=weights)

    batch_logits = model(train_tokens)
    loss = loss_fn(batch_logits, labels)
    print(f"   Loss: {loss.item():.4f}")

    print("\n" + "=" * 50)
    print("✅ Tous les tests passent!")
