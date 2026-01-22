"""
V14/V15 Cytology — MLP Classification Head avec Fusion Multimodale

Architecture optimisée pour fusionner:
- H-Optimus-0 embeddings (1536D)
- Morphometric features (20D)

Fusion Strategies:
- V14: Concat simple (embedding | morpho)
- V15: Gated Feature Fusion (GFF) — Pondération adaptative

Principe Critique: BatchNormalization sur features morphométriques pour
équilibrer les gradients (1536 dims >> 20 dims).

Author: V14/V15 Cytology Branch
Date: 2026-01-19 (V14), 2026-01-22 (V15 GFF)
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
#  V15: GATED FEATURE FUSION (GFF)
# ═════════════════════════════════════════════════════════════════════════════

class GatedFeatureFusion(nn.Module):
    """
    Gated Feature Fusion (GFF) pour fusion adaptative Visual/Morphométrique

    Principe: Le gate apprend automatiquement à pondérer les features visuelles
    vs morphométriques selon le contexte de chaque cellule.

    Formule:
        g = σ(W_g · [f_visual; f_morpho_proj] + b_g)
        f_fused = g ⊙ f_visual + (1-g) ⊙ f_morpho_proj

    Avantages vs Concat Simple (V14):
        - Pondération adaptative par sample
        - Meilleure gestion du déséquilibre dimensionnel (1536 >> 20)
        - Interprétabilité (gate values = importance relative)

    Usage:
        gff = GatedFeatureFusion(embedding_dim=1536, morpho_dim=20)
        fused = gff(visual_features, morpho_features)  # (B, 1536)
    """

    def __init__(
        self,
        embedding_dim: int = 1536,
        morpho_dim: int = 20,
        use_batchnorm_morpho: bool = True
    ):
        """
        Args:
            embedding_dim: Dimension embedding H-Optimus (1536)
            morpho_dim: Dimension features morphométriques (20)
            use_batchnorm_morpho: Activer BatchNorm sur morpho (RECOMMANDÉ)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.morpho_dim = morpho_dim

        # ═══════════════════════════════════════════════════════════════════
        #  NORMALISATION MORPHO (CRITIQUE — équilibrage gradients)
        # ═══════════════════════════════════════════════════════════════════

        if use_batchnorm_morpho:
            self.morpho_batchnorm = nn.BatchNorm1d(morpho_dim)
        else:
            self.morpho_batchnorm = nn.Identity()

        # ═══════════════════════════════════════════════════════════════════
        #  PROJECTION MORPHO → Embedding Space
        # ═══════════════════════════════════════════════════════════════════

        # Projeter morpho (20D) vers même espace que visual (1536D)
        self.morpho_projection = nn.Sequential(
            nn.Linear(morpho_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

        # ═══════════════════════════════════════════════════════════════════
        #  GATE NETWORK
        # ═══════════════════════════════════════════════════════════════════

        # Gate prend [visual; morpho_proj] (3072D) → gate (1536D)
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )

        # Initialisation
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialisation pour gate neutre (≈0.5) au début"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Biais du gate à 0 pour sortie initiale ≈ 0.5 (neutre)
        # Déjà fait par init constant_(m.bias, 0)

    def forward(
        self,
        visual_features: torch.Tensor,
        morpho_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass avec fusion gated

        Args:
            visual_features: H-Optimus embeddings (B, 1536)
            morpho_features: Morphometric features (B, 20)

        Returns:
            fused: Features fusionnées (B, 1536)
            gate: Valeurs du gate pour interprétabilité (B, 1536)
        """
        # Validation shapes
        assert visual_features.size(1) == self.embedding_dim, \
            f"Visual dim mismatch: {visual_features.size(1)} vs {self.embedding_dim}"
        assert morpho_features.size(1) == self.morpho_dim, \
            f"Morpho dim mismatch: {morpho_features.size(1)} vs {self.morpho_dim}"

        # ═══════════════════════════════════════════════════════════════════
        #  NORMALISATION + PROJECTION MORPHO
        # ═══════════════════════════════════════════════════════════════════

        morpho_normalized = self.morpho_batchnorm(morpho_features)
        morpho_projected = self.morpho_projection(morpho_normalized)  # (B, 1536)

        # ═══════════════════════════════════════════════════════════════════
        #  GATED FUSION
        # ═══════════════════════════════════════════════════════════════════

        # Concat pour gate input
        gate_input = torch.cat([visual_features, morpho_projected], dim=1)  # (B, 3072)

        # Gate: σ(W_g · [f_visual; f_morpho_proj] + b_g)
        gate = self.gate_network(gate_input)  # (B, 1536)

        # Fusion: g ⊙ f_visual + (1-g) ⊙ f_morpho_proj
        fused = gate * visual_features + (1 - gate) * morpho_projected  # (B, 1536)

        return fused, gate

    def get_gate_statistics(self, gate: torch.Tensor) -> dict:
        """
        Statistiques du gate pour interprétabilité

        Args:
            gate: Valeurs du gate (B, 1536)

        Returns:
            dict avec mean, std, visual_weight (mean gate = poids visual)
        """
        with torch.no_grad():
            return {
                'gate_mean': gate.mean().item(),
                'gate_std': gate.std().item(),
                'visual_weight': gate.mean().item(),  # gate proche 1 = plus de visual
                'morpho_weight': 1 - gate.mean().item()  # gate proche 0 = plus de morpho
            }


class CytologyClassifierV15(nn.Module):
    """
    V15 Cytology Classifier avec Gated Feature Fusion

    Architecture:
        Input: embedding (1536D) + morpho (20D)
        → GatedFeatureFusion → fused (1536D)
        → Dense(512) + ReLU + Dropout(0.3)
        → Dense(256) + ReLU + Dropout(0.2)
        → Dense(num_classes)

    Différences vs V14 (CytologyClassifier):
        - V14: Concat simple [embedding; morpho] = 1556D → MLP
        - V15: GFF [embedding; morpho] → fused 1536D → MLP

    Usage:
        model = CytologyClassifierV15(num_classes=6)  # APCData Bethesda
        logits, gate = model(embeddings, morpho_features)
    """

    def __init__(
        self,
        embedding_dim: int = 1536,
        morpho_dim: int = 20,
        hidden_dims: Tuple[int, int] = (512, 256),
        num_classes: int = 6,  # APCData: 6 classes Bethesda
        dropout_rates: Tuple[float, float] = (0.3, 0.2),
        use_batchnorm_morpho: bool = True
    ):
        """
        Args:
            embedding_dim: Dimension embedding H-Optimus (1536)
            morpho_dim: Dimension features morphométriques (20)
            hidden_dims: Dimensions couches cachées (512, 256)
            num_classes: Nombre de classes (6 pour APCData, 7 pour SIPaKMeD)
            dropout_rates: Taux dropout (0.3, 0.2)
            use_batchnorm_morpho: Activer BatchNorm sur morpho dans GFF
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.morpho_dim = morpho_dim
        self.num_classes = num_classes

        # ═══════════════════════════════════════════════════════════════════
        #  GATED FEATURE FUSION
        # ═══════════════════════════════════════════════════════════════════

        self.gff = GatedFeatureFusion(
            embedding_dim=embedding_dim,
            morpho_dim=morpho_dim,
            use_batchnorm_morpho=use_batchnorm_morpho
        )

        # ═══════════════════════════════════════════════════════════════════
        #  CLASSIFICATION HEAD (prend fused 1536D, pas 1556D)
        # ═══════════════════════════════════════════════════════════════════

        # Couche 1: 1536 → 512
        self.fc1 = nn.Linear(embedding_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rates[0])

        # Couche 2: 512 → 256
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rates[1])

        # Couche finale: 256 → num_classes
        self.fc_out = nn.Linear(hidden_dims[1], num_classes)

        # Initialisation poids
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialisation Xavier pour classification head"""
        for name, m in self.named_modules():
            # Skip GFF (déjà initialisé)
            if name.startswith('gff'):
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        embedding: torch.Tensor,
        morpho_features: torch.Tensor,
        return_gate: bool = False
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            embedding: H-Optimus embeddings (B, 1536)
            morpho_features: Morphometric features (B, 20)
            return_gate: Si True, retourne aussi les valeurs du gate

        Returns:
            logits: (B, num_classes) — Logits AVANT softmax
            gate: (B, 1536) — Seulement si return_gate=True
        """
        # ═══════════════════════════════════════════════════════════════════
        #  GATED FEATURE FUSION
        # ═══════════════════════════════════════════════════════════════════

        fused, gate = self.gff(embedding, morpho_features)  # (B, 1536)

        # ═══════════════════════════════════════════════════════════════════
        #  CLASSIFICATION HEAD
        # ═══════════════════════════════════════════════════════════════════

        x = self.fc1(fused)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        logits = self.fc_out(x)

        if return_gate:
            return logits, gate
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
        self.eval()
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
        Retourne l'embedding fusionné AVANT classification (pour visualisation t-SNE)

        Args:
            embedding: (B, 1536)
            morpho_features: (B, 20)

        Returns:
            latent_embedding: (B, 256) — Après fc2
        """
        fused, _ = self.gff(embedding, morpho_features)

        x = self.fc1(fused)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        return x

    def get_gate_analysis(
        self,
        embedding: torch.Tensor,
        morpho_features: torch.Tensor
    ) -> dict:
        """
        Analyse du comportement du gate (interprétabilité)

        Returns:
            dict avec statistiques gate et importance relative visual/morpho
        """
        self.eval()
        with torch.no_grad():
            _, gate = self.gff(embedding, morpho_features)
            return self.gff.get_gate_statistics(gate)


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
    """Test V14 et V15 architectures"""

    batch_size = 4
    embedding = torch.randn(batch_size, 1536)
    morpho = torch.randn(batch_size, 20)

    # ═════════════════════════════════════════════════════════════════════
    #  TEST V14 CLASSIFIER (Concat Simple)
    # ═════════════════════════════════════════════════════════════════════

    print("=" * 80)
    print("V14 CYTOLOGY CLASSIFIER (Concat Simple)")
    print("=" * 80)

    model_v14 = CytologyClassifier(num_classes=7)  # SIPaKMeD
    print(f"Total Parameters: {count_parameters(model_v14):,}")
    print(f"Architecture: 1556 → 512 → 256 → 7")
    print(f"Fusion: Concat [embedding | morpho]")

    model_v14.train()
    logits_v14 = model_v14(embedding, morpho)
    print(f"Logits shape: {logits_v14.shape}")

    # ═════════════════════════════════════════════════════════════════════
    #  TEST V15 CLASSIFIER (Gated Feature Fusion)
    # ═════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("V15 CYTOLOGY CLASSIFIER (Gated Feature Fusion)")
    print("=" * 80)

    model_v15 = CytologyClassifierV15(num_classes=6)  # APCData Bethesda
    print(f"Total Parameters: {count_parameters(model_v15):,}")
    print(f"Architecture: GFF(1536,20) → 1536 → 512 → 256 → 6")
    print(f"Fusion: Gated Feature Fusion")

    model_v15.train()
    logits_v15, gate = model_v15(embedding, morpho, return_gate=True)
    print(f"Logits shape: {logits_v15.shape}")
    print(f"Gate shape: {gate.shape}")

    # Analyse gate
    model_v15.eval()
    gate_stats = model_v15.get_gate_analysis(embedding, morpho)
    print(f"\nGate Analysis:")
    print(f"  Mean: {gate_stats['gate_mean']:.4f}")
    print(f"  Std: {gate_stats['gate_std']:.4f}")
    print(f"  Visual Weight: {gate_stats['visual_weight']:.2%}")
    print(f"  Morpho Weight: {gate_stats['morpho_weight']:.2%}")

    # ═════════════════════════════════════════════════════════════════════
    #  TEST FOCAL LOSS
    # ═════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("FOCAL LOSS TEST")
    print("=" * 80)

    targets_v14 = torch.tensor([0, 2, 4, 6])
    targets_v15 = torch.tensor([0, 2, 4, 5])  # 6 classes max

    criterion = FocalLoss(gamma=2.0)
    loss_v14 = criterion(logits_v14, targets_v14)
    loss_v15 = criterion(logits_v15, targets_v15)

    print(f"Focal Loss V14: {loss_v14.item():.4f}")
    print(f"Focal Loss V15: {loss_v15.item():.4f}")

    # ═════════════════════════════════════════════════════════════════════
    #  COMPARAISON PARAMÈTRES
    # ═════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON")
    print("=" * 80)

    params_v14 = count_parameters(model_v14)
    params_v15 = count_parameters(model_v15)

    print(f"V14 (Concat): {params_v14:,} params")
    print(f"V15 (GFF):    {params_v15:,} params")
    print(f"Overhead GFF: +{params_v15 - params_v14:,} params (+{(params_v15/params_v14 - 1)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("✅ V14 & V15 Architectures validated — Ready for training")
    print("=" * 80)
