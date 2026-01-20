"""
V14 Cytology — MLP Classification Head avec Fusion Multimodale

Architecture optimisée pour fusionner:
- H-Optimus-0 embeddings (1536D)
- Morphometric features (20D)

Principe Critique: BatchNormalization sur features morphométriques pour
équilibrer les gradients (1536 dims >> 20 dims).

Author: V14 Cytology Branch
Date: 2026-01-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CytologyClassifier(nn.Module):
    """
    MLP Classification Head pour Cytologie V14

    Architecture:
        Input: embedding (1536D) + morpho (20D) = 1556D
        → Dense(512) + ReLU + Dropout(0.3)
        → Dense(256) + ReLU + Dropout(0.2)
        → Dense(num_classes) + Softmax

    Features Critiques:
        - BatchNorm sur morpho features (équilibrage gradients)
        - Dropout progressif (0.3 → 0.2) pour régularisation
        - Support dual-input (embedding + morpho séparés)

    Usage:
        model = CytologyClassifier(num_classes=7)  # SIPaKMeD
        logits = model(embeddings, morpho_features)
        probs = F.softmax(logits, dim=1)
    """

    def __init__(
        self,
        embedding_dim: int = 1536,
        morpho_dim: int = 20,
        hidden_dims: Tuple[int, int] = (512, 256),
        num_classes: int = 7,
        dropout_rates: Tuple[float, float] = (0.3, 0.2),
        use_batchnorm_morpho: bool = True
    ):
        """
        Args:
            embedding_dim: Dimension embedding H-Optimus (1536)
            morpho_dim: Dimension features morphométriques (20)
            hidden_dims: Dimensions couches cachées (512, 256)
            num_classes: Nombre de classes (7 pour SIPaKMeD)
            dropout_rates: Taux dropout (0.3, 0.2)
            use_batchnorm_morpho: Activer BatchNorm sur morpho (RECOMMANDÉ)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.morpho_dim = morpho_dim
        self.num_classes = num_classes
        self.use_batchnorm_morpho = use_batchnorm_morpho

        # ═════════════════════════════════════════════════════════════════════
        #  NORMALISATION MORPHO FEATURES (CRITIQUE)
        # ═════════════════════════════════════════════════════════════════════

        if use_batchnorm_morpho:
            self.morpho_batchnorm = nn.BatchNorm1d(morpho_dim)
        else:
            self.morpho_batchnorm = nn.Identity()

        # ═════════════════════════════════════════════════════════════════════
        #  FUSION + CLASSIFICATION HEAD
        # ═════════════════════════════════════════════════════════════════════

        fusion_dim = embedding_dim + morpho_dim  # 1556

        # Couche 1: 1556 → 512
        self.fc1 = nn.Linear(fusion_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rates[0])

        # Couche 2: 512 → 256
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rates[1])

        # Couche finale: 256 → num_classes
        self.fc_out = nn.Linear(hidden_dims[1], num_classes)

        # Initialisation poids (Xavier uniform)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialisation Xavier pour améliorer convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        embedding: torch.Tensor,
        morpho_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            embedding: H-Optimus embeddings (B, 1536)
            morpho_features: Morphometric features (B, 20)

        Returns:
            logits: (B, num_classes) — Logits AVANT softmax

        Notes:
            - Softmax appliqué externally (loss ou inference)
            - BatchNorm en mode training/eval automatique via model.train()/eval()
        """
        # Validation shapes
        batch_size = embedding.size(0)
        assert embedding.size(1) == self.embedding_dim, \
            f"Embedding dim mismatch: {embedding.size(1)} vs {self.embedding_dim}"
        assert morpho_features.size(1) == self.morpho_dim, \
            f"Morpho dim mismatch: {morpho_features.size(1)} vs {self.morpho_dim}"

        # ═════════════════════════════════════════════════════════════════════
        #  NORMALISATION MORPHO (CRITIQUE)
        # ═════════════════════════════════════════════════════════════════════

        # BatchNorm sur morpho pour équilibrer avec embedding
        # Embedding déjà normalisé par H-Optimus (mean≈0, std≈1)
        # Morpho: valeurs brutes (area=500, nc_ratio=0.7, etc.)
        morpho_normalized = self.morpho_batchnorm(morpho_features)

        # ═════════════════════════════════════════════════════════════════════
        #  FUSION MULTIMODALE
        # ═════════════════════════════════════════════════════════════════════

        # Concaténation [embedding (1536) | morpho (20)] → (B, 1556)
        fused = torch.cat([embedding, morpho_normalized], dim=1)

        # ═════════════════════════════════════════════════════════════════════
        #  CLASSIFICATION HEAD
        # ═════════════════════════════════════════════════════════════════════

        # Couche 1
        x = self.fc1(fused)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Couche 2
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Logits
        logits = self.fc_out(x)

        return logits

    def predict_proba(
        self,
        embedding: torch.Tensor,
        morpho_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Prédiction avec softmax (inference)

        Args:
            embedding: (B, 1536)
            morpho_features: (B, 20)

        Returns:
            probs: (B, num_classes) — Probabilités [0, 1]
        """
        self.eval()  # Mode eval (BatchNorm utilise running stats)
        with torch.no_grad():
            logits = self.forward(embedding, morpho_features)
            probs = F.softmax(logits, dim=1)
        return probs

    def get_embedding_before_classification(
        self,
        embedding: torch.Tensor,
        morpho_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Retourne l'embedding fusionné AVANT classification (pour visualisation)

        Args:
            embedding: (B, 1536)
            morpho_features: (B, 20)

        Returns:
            fused_embedding: (B, 256) — Après fc2 (embedding latent)
        """
        morpho_normalized = self.morpho_batchnorm(morpho_features)
        fused = torch.cat([embedding, morpho_normalized], dim=1)

        x = self.fc1(fused)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        # Pas de dropout ici (voulons embedding propre)

        return x


# ═════════════════════════════════════════════════════════════════════════════
#  LOSS FUNCTION (Class Imbalance Aware)
# ═════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss pour gérer déséquilibre des classes (Lin et al., 2017)

    Principe: Réduit le poids des exemples faciles (bien classés),
              focus sur exemples difficiles (mal classés)

    Formule: FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    Args:
        alpha: Poids par classe (None = pas de pondération)
        gamma: Facteur de focalisation (2.0 = standard)
               gamma=0 → CrossEntropy standard
               gamma>0 → Réduit influence easy examples
        reduction: 'mean' | 'sum' | 'none'

    Usage:
        # SIPaKMeD: 7 classes déséquilibrées
        # Normal: ~45%, Malignant: ~10%
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        loss = criterion(logits, targets)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) — Logits AVANT softmax
            targets: (B,) — Labels entiers [0, num_classes-1]

        Returns:
            loss: Scalar (si reduction='mean')
        """
        # Softmax
        probs = F.softmax(logits, dim=1)

        # Probabilité de la vraie classe
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1))
        p_t = (probs * targets_one_hot).sum(dim=1)

        # Focal term
        focal_weight = (1 - p_t) ** self.gamma

        # Cross-entropy
        ce_loss = -torch.log(p_t + 1e-8)

        # Focal loss
        loss = focal_weight * ce_loss

        # Alpha (pondération par classe)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ═════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def compute_class_weights(class_counts: torch.Tensor) -> torch.Tensor:
    """
    Calcule poids par classe (inverse fréquence)

    Args:
        class_counts: Tensor (num_classes,) avec nombre samples par classe

    Returns:
        weights: Tensor (num_classes,) normalisé

    Example:
        # SIPaKMeD: [787, 518, 502, 1484, 793, 1470, 813]
        # Normal: 1807, Abnormal: 2242
        counts = torch.tensor([787, 518, 502, 1484, 793, 1470, 813])
        weights = compute_class_weights(counts)
        # Output: Weights inversement proportionnels
    """
    total = class_counts.sum()
    weights = total / (len(class_counts) * class_counts.float())
    weights = weights / weights.sum() * len(class_counts)  # Normaliser
    return weights


def count_parameters(model: nn.Module) -> int:
    """Compte nombre de paramètres trainables"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    """Test architecture"""

    # Créer modèle
    model = CytologyClassifier(num_classes=7)

    print("=" * 80)
    print("V14 CYTOLOGY CLASSIFIER")
    print("=" * 80)
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"Architecture: 1556 → 512 → 256 → 7")
    print(f"BatchNorm Morpho: {model.use_batchnorm_morpho}")
    print("=" * 80)

    # Test forward pass
    batch_size = 4
    embedding = torch.randn(batch_size, 1536)
    morpho = torch.randn(batch_size, 20)

    # Mode training
    model.train()
    logits_train = model(embedding, morpho)
    print(f"\nTrain mode logits shape: {logits_train.shape}")

    # Mode eval
    model.eval()
    probs = model.predict_proba(embedding, morpho)
    print(f"Eval mode probs shape: {probs.shape}")
    print(f"Probs sum (should be 1.0): {probs.sum(dim=1)}")

    # Test Focal Loss
    targets = torch.tensor([0, 2, 4, 6])
    criterion = FocalLoss(gamma=2.0)
    loss = criterion(logits_train, targets)
    print(f"\nFocal Loss: {loss.item():.4f}")

    print("\n✅ Architecture validated — Ready for training")
