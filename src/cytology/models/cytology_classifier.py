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


# ═════════════════════════════════════════════════════════════════════════════
#  V15.2: MULTI-HEAD HIERARCHICAL CLASSIFIER
# ═════════════════════════════════════════════════════════════════════════════

# Bethesda System Classification
BETHESDA_CLASSES = {
    0: "NILM",   # Negative for Intraepithelial Lesion or Malignancy
    1: "ASCUS",  # Atypical Squamous Cells of Undetermined Significance
    2: "ASCH",   # Atypical Squamous Cells, cannot exclude HSIL
    3: "LSIL",   # Low-grade Squamous Intraepithelial Lesion
    4: "HSIL",   # High-grade Squamous Intraepithelial Lesion
    5: "SCC"     # Squamous Cell Carcinoma
}

# Binary mapping: Normal (NILM) vs Abnormal (all others)
BINARY_MAPPING = {
    0: 0,  # NILM → Normal
    1: 1,  # ASCUS → Abnormal
    2: 1,  # ASCH → Abnormal
    3: 1,  # LSIL → Abnormal
    4: 1,  # HSIL → Abnormal
    5: 1   # SCC → Abnormal
}

# Severity mapping: Low-risk (NILM, ASCUS, LSIL) vs High-risk (ASCH, HSIL, SCC)
SEVERITY_MAPPING = {
    0: 0,  # NILM → Low-risk
    1: 0,  # ASCUS → Low-risk
    2: 1,  # ASCH → High-risk
    3: 0,  # LSIL → Low-risk
    4: 1,  # HSIL → High-risk
    5: 1   # SCC → High-risk
}


class RejectionLayer(nn.Module):
    """
    Rejection Layer pour cas incertains (Conformal Prediction approach)

    Principe: Apprend à prédire si une cellule doit être envoyée en révision
    manuelle basé sur l'incertitude des prédictions.

    Méthode: Utilise la Non-Conformity Score basée sur:
        1. Entropie de la distribution de probabilité
        2. Différence entre top-1 et top-2 probabilités
        3. Features latentes (haute dimension = pattern complexe)

    Usage:
        rejection_layer = RejectionLayer(latent_dim=256)
        should_reject, confidence = rejection_layer(latent_features, probs)
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 64,
        rejection_threshold: float = 0.5
    ):
        """
        Args:
            latent_dim: Dimension des features latentes (256)
            hidden_dim: Dimension couche cachée (64)
            rejection_threshold: Seuil de rejet (0.5 = par défaut)
        """
        super().__init__()

        self.rejection_threshold = rejection_threshold

        # Network pour prédire rejection probability
        # Input: latent (256) + uncertainty_features (3)
        self.rejection_network = nn.Sequential(
            nn.Linear(latent_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_uncertainty_features(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Calcule features d'incertitude à partir des probabilités

        Args:
            probs: (B, num_classes) — Probabilités après softmax

        Returns:
            uncertainty_features: (B, 3)
                - Entropy normalisée
                - Margin (top1 - top2)
                - Max probability
        """
        # Entropy: H = -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
        max_entropy = torch.log(torch.tensor(probs.size(1), dtype=torch.float, device=probs.device))
        normalized_entropy = entropy / max_entropy

        # Margin: top1 - top2
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        margin = (sorted_probs[:, 0:1] - sorted_probs[:, 1:2])

        # Max probability
        max_prob = sorted_probs[:, 0:1]

        return torch.cat([normalized_entropy, margin, max_prob], dim=1)

    def forward(
        self,
        latent_features: torch.Tensor,
        probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            latent_features: (B, latent_dim) — Features avant classification
            probs: (B, num_classes) — Probabilités après softmax

        Returns:
            should_reject: (B, 1) — Bool tensor (True = envoyer en révision)
            rejection_prob: (B, 1) — Probabilité de rejet [0, 1]
        """
        # Calcul features d'incertitude
        uncertainty_features = self.compute_uncertainty_features(probs)

        # Concat latent + uncertainty
        rejection_input = torch.cat([latent_features, uncertainty_features], dim=1)

        # Prédiction
        rejection_prob = self.rejection_network(rejection_input)

        # Décision binaire
        should_reject = rejection_prob > self.rejection_threshold

        return should_reject, rejection_prob

    def set_threshold(self, threshold: float):
        """Ajuste le seuil de rejet (utile pour calibration)"""
        self.rejection_threshold = threshold


class CytologyMultiHead(nn.Module):
    """
    V15.2 Multi-Head Hierarchical Classifier pour Cytologie

    Architecture hiérarchique validée par expert (industrie: Hologic, BD-Techcyte):

        Head 1 (Binary): Normal (NILM) vs Abnormal (ASCUS→SCC)
            └─ Si Abnormal →

        Head 2 (Severity): Low-risk (ASCUS, LSIL) vs High-risk (ASCH, HSIL, SCC)
            └─ Input: GFF features + morpho features (expert recommendation)

        Head 3 (Fine-grained): 6 classes Bethesda complètes

        Rejection Layer: Identifie cas incertains pour révision manuelle

    Avantages:
        - Décision médicale progressive (triage → diagnostic)
        - Interprétabilité (chaque head = question clinique)
        - Rejection layer = safety net (jamais rater un cancer)

    Usage:
        model = CytologyMultiHead(embedding_dim=1536, morpho_dim=20)
        outputs = model(embeddings, morpho_features)
        # outputs = {
        #     'binary_logits': (B, 2),
        #     'severity_logits': (B, 2),
        #     'fine_logits': (B, 6),
        #     'binary_probs': (B, 2),
        #     'severity_probs': (B, 2),
        #     'fine_probs': (B, 6),
        #     'should_reject': (B, 1),
        #     'rejection_prob': (B, 1),
        #     'gate': (B, 1536)
        # }
    """

    def __init__(
        self,
        embedding_dim: int = 1536,
        morpho_dim: int = 20,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
        use_batchnorm_morpho: bool = True,
        rejection_threshold: float = 0.5
    ):
        """
        Args:
            embedding_dim: Dimension embedding H-Optimus (1536)
            morpho_dim: Dimension features morphométriques (20)
            hidden_dim: Dimension couche cachée partagée (256)
            dropout_rate: Taux dropout (0.3)
            use_batchnorm_morpho: Activer BatchNorm sur morpho dans GFF
            rejection_threshold: Seuil pour rejection layer (0.5)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.morpho_dim = morpho_dim
        self.hidden_dim = hidden_dim

        # ═══════════════════════════════════════════════════════════════════
        #  GATED FEATURE FUSION (Shared Backbone)
        # ═══════════════════════════════════════════════════════════════════

        self.gff = GatedFeatureFusion(
            embedding_dim=embedding_dim,
            morpho_dim=morpho_dim,
            use_batchnorm_morpho=use_batchnorm_morpho
        )

        # ═══════════════════════════════════════════════════════════════════
        #  SHARED FEATURE EXTRACTOR
        # ═══════════════════════════════════════════════════════════════════

        # Shared: GFF output (1536) → latent (256)
        self.shared_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.66)  # 0.2
        )

        # ═══════════════════════════════════════════════════════════════════
        #  HEAD 1: BINARY (Normal vs Abnormal)
        # ═══════════════════════════════════════════════════════════════════

        # Input: latent (256)
        self.head_binary = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # Normal, Abnormal
        )

        # ═══════════════════════════════════════════════════════════════════
        #  HEAD 2: SEVERITY (Low-risk vs High-risk)
        # ═══════════════════════════════════════════════════════════════════

        # Input: latent (256) + morpho_normalized (20)
        # Expert recommendation: morpho features help distinguish severity
        # (nucleus area, chromatin granularity, shape irregularity)
        if use_batchnorm_morpho:
            self.morpho_batchnorm_severity = nn.BatchNorm1d(morpho_dim)
        else:
            self.morpho_batchnorm_severity = nn.Identity()

        self.head_severity = nn.Sequential(
            nn.Linear(hidden_dim + morpho_dim, 64),  # 256 + 20 = 276
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # Low-risk, High-risk
        )

        # ═══════════════════════════════════════════════════════════════════
        #  HEAD 3: FINE-GRAINED (6 Bethesda Classes)
        # ═══════════════════════════════════════════════════════════════════

        # Input: latent (256)
        self.head_fine = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 6)  # NILM, ASCUS, ASCH, LSIL, HSIL, SCC
        )

        # ═══════════════════════════════════════════════════════════════════
        #  REJECTION LAYER
        # ═══════════════════════════════════════════════════════════════════

        self.rejection_layer = RejectionLayer(
            latent_dim=hidden_dim,
            hidden_dim=64,
            rejection_threshold=rejection_threshold
        )

        # Initialisation
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialisation Xavier pour tous les heads"""
        for name, m in self.named_modules():
            # Skip GFF et rejection_layer (déjà initialisés)
            if name.startswith('gff') or name.startswith('rejection_layer'):
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        embedding: torch.Tensor,
        morpho_features: torch.Tensor,
        return_all: bool = True
    ) -> dict:
        """
        Forward pass multi-head

        Args:
            embedding: H-Optimus embeddings (B, 1536)
            morpho_features: Morphometric features (B, 20)
            return_all: Si True, retourne tous les outputs

        Returns:
            dict avec:
                - binary_logits: (B, 2)
                - severity_logits: (B, 2)
                - fine_logits: (B, 6)
                - binary_probs: (B, 2) — Si return_all
                - severity_probs: (B, 2) — Si return_all
                - fine_probs: (B, 6) — Si return_all
                - should_reject: (B, 1) — Si return_all
                - rejection_prob: (B, 1) — Si return_all
                - gate: (B, 1536) — Si return_all
                - latent: (B, 256) — Si return_all
        """
        # ═══════════════════════════════════════════════════════════════════
        #  GATED FEATURE FUSION
        # ═══════════════════════════════════════════════════════════════════

        fused, gate = self.gff(embedding, morpho_features)  # (B, 1536)

        # ═══════════════════════════════════════════════════════════════════
        #  SHARED ENCODER
        # ═══════════════════════════════════════════════════════════════════

        latent = self.shared_encoder(fused)  # (B, 256)

        # ═══════════════════════════════════════════════════════════════════
        #  HEAD 1: BINARY
        # ═══════════════════════════════════════════════════════════════════

        binary_logits = self.head_binary(latent)  # (B, 2)

        # ═══════════════════════════════════════════════════════════════════
        #  HEAD 2: SEVERITY (avec morpho features)
        # ═══════════════════════════════════════════════════════════════════

        morpho_norm_severity = self.morpho_batchnorm_severity(morpho_features)
        severity_input = torch.cat([latent, morpho_norm_severity], dim=1)  # (B, 276)
        severity_logits = self.head_severity(severity_input)  # (B, 2)

        # ═══════════════════════════════════════════════════════════════════
        #  HEAD 3: FINE-GRAINED
        # ═══════════════════════════════════════════════════════════════════

        fine_logits = self.head_fine(latent)  # (B, 6)

        # ═══════════════════════════════════════════════════════════════════
        #  OUTPUT
        # ═══════════════════════════════════════════════════════════════════

        outputs = {
            'binary_logits': binary_logits,
            'severity_logits': severity_logits,
            'fine_logits': fine_logits
        }

        if return_all:
            # Probabilities
            binary_probs = F.softmax(binary_logits, dim=1)
            severity_probs = F.softmax(severity_logits, dim=1)
            fine_probs = F.softmax(fine_logits, dim=1)

            outputs['binary_probs'] = binary_probs
            outputs['severity_probs'] = severity_probs
            outputs['fine_probs'] = fine_probs

            # Rejection (basé sur fine-grained probs)
            should_reject, rejection_prob = self.rejection_layer(latent, fine_probs)
            outputs['should_reject'] = should_reject
            outputs['rejection_prob'] = rejection_prob

            # Debug/interpretability
            outputs['gate'] = gate
            outputs['latent'] = latent

        return outputs

    def predict(
        self,
        embedding: torch.Tensor,
        morpho_features: torch.Tensor
    ) -> dict:
        """
        Prédiction avec labels et confidences

        Returns:
            dict avec:
                - binary_pred: (B,) — 0=Normal, 1=Abnormal
                - binary_conf: (B,) — Confiance [0, 1]
                - severity_pred: (B,) — 0=Low-risk, 1=High-risk
                - severity_conf: (B,) — Confiance [0, 1]
                - fine_pred: (B,) — 0-5 (Bethesda)
                - fine_conf: (B,) — Confiance [0, 1]
                - should_reject: (B,) — Bool
                - rejection_prob: (B,) — [0, 1]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(embedding, morpho_features, return_all=True)

            # Binary
            binary_conf, binary_pred = outputs['binary_probs'].max(dim=1)

            # Severity
            severity_conf, severity_pred = outputs['severity_probs'].max(dim=1)

            # Fine-grained
            fine_conf, fine_pred = outputs['fine_probs'].max(dim=1)

            return {
                'binary_pred': binary_pred,
                'binary_conf': binary_conf,
                'severity_pred': severity_pred,
                'severity_conf': severity_conf,
                'fine_pred': fine_pred,
                'fine_conf': fine_conf,
                'should_reject': outputs['should_reject'].squeeze(-1),
                'rejection_prob': outputs['rejection_prob'].squeeze(-1)
            }

    def get_hierarchical_prediction(
        self,
        embedding: torch.Tensor,
        morpho_features: torch.Tensor
    ) -> dict:
        """
        Prédiction hiérarchique progressive (triage clinique)

        Flow:
            1. Binary: Normal? → Si oui, stop
            2. Severity: High-risk? → Priorisation
            3. Fine: Classe précise
            4. Rejection: Besoin révision?

        Returns:
            dict avec interprétation clinique
        """
        preds = self.predict(embedding, morpho_features)
        batch_size = embedding.size(0)

        results = []
        for i in range(batch_size):
            result = {
                'is_normal': bool(preds['binary_pred'][i] == 0),
                'normal_confidence': float(preds['binary_conf'][i]) if preds['binary_pred'][i] == 0 else float(1 - preds['binary_conf'][i]),
                'is_high_risk': bool(preds['severity_pred'][i] == 1),
                'severity_confidence': float(preds['severity_conf'][i]),
                'bethesda_class': int(preds['fine_pred'][i]),
                'bethesda_name': BETHESDA_CLASSES[int(preds['fine_pred'][i])],
                'bethesda_confidence': float(preds['fine_conf'][i]),
                'needs_review': bool(preds['should_reject'][i]),
                'review_probability': float(preds['rejection_prob'][i])
            }

            # Clinical interpretation
            if result['is_normal']:
                result['clinical_action'] = "Routine follow-up"
                result['priority'] = "LOW"
            elif result['is_high_risk']:
                result['clinical_action'] = "Immediate colposcopy recommended"
                result['priority'] = "HIGH"
            else:
                result['clinical_action'] = "HPV testing / 6-month follow-up"
                result['priority'] = "MEDIUM"

            if result['needs_review']:
                result['clinical_action'] += " (MANUAL REVIEW REQUIRED)"
                result['priority'] = "REVIEW"

            results.append(result)

        return results

    def set_rejection_threshold(self, threshold: float):
        """Ajuste le seuil de rejet"""
        self.rejection_layer.set_threshold(threshold)


class MultiHeadLoss(nn.Module):
    """
    Loss combinée pour CytologyMultiHead

    Combine:
        - Binary loss (CrossEntropy ou Focal)
        - Severity loss (CrossEntropy ou Focal)
        - Fine-grained loss (CrossEntropy ou Focal)

    Pondération par importance clinique:
        - Binary: λ=1.0 (triage)
        - Severity: λ=1.5 (priorisation clinique)
        - Fine: λ=1.0 (diagnostic précis)
    """

    def __init__(
        self,
        lambda_binary: float = 1.0,
        lambda_severity: float = 1.5,
        lambda_fine: float = 1.0,
        use_focal: bool = True,
        gamma: float = 2.0,
        class_weights_fine: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.lambda_binary = lambda_binary
        self.lambda_severity = lambda_severity
        self.lambda_fine = lambda_fine

        if use_focal:
            self.loss_binary = FocalLoss(gamma=gamma)
            self.loss_severity = FocalLoss(gamma=gamma)
            self.loss_fine = FocalLoss(alpha=class_weights_fine, gamma=gamma)
        else:
            self.loss_binary = nn.CrossEntropyLoss()
            self.loss_severity = nn.CrossEntropyLoss()
            if class_weights_fine is not None:
                self.loss_fine = nn.CrossEntropyLoss(weight=class_weights_fine)
            else:
                self.loss_fine = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: dict,
        targets_fine: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calcul loss combinée

        Args:
            outputs: dict from CytologyMultiHead.forward()
            targets_fine: (B,) — Labels Bethesda [0-5]

        Returns:
            total_loss: Scalar
            loss_dict: dict avec losses individuelles
        """
        # Dériver targets binary et severity depuis fine
        targets_binary = torch.tensor(
            [BINARY_MAPPING[t.item()] for t in targets_fine],
            device=targets_fine.device,
            dtype=torch.long
        )
        targets_severity = torch.tensor(
            [SEVERITY_MAPPING[t.item()] for t in targets_fine],
            device=targets_fine.device,
            dtype=torch.long
        )

        # Losses individuelles
        loss_binary = self.loss_binary(outputs['binary_logits'], targets_binary)
        loss_severity = self.loss_severity(outputs['severity_logits'], targets_severity)
        loss_fine = self.loss_fine(outputs['fine_logits'], targets_fine)

        # Total pondéré
        total_loss = (
            self.lambda_binary * loss_binary +
            self.lambda_severity * loss_severity +
            self.lambda_fine * loss_fine
        )

        loss_dict = {
            'loss_binary': loss_binary.item(),
            'loss_severity': loss_severity.item(),
            'loss_fine': loss_fine.item(),
            'loss_total': total_loss.item()
        }

        return total_loss, loss_dict


if __name__ == '__main__':
    """Test V14, V15 et V15.2 (MultiHead) architectures"""

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
    #  TEST V15.2 MULTI-HEAD CLASSIFIER
    # ═════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("V15.2 MULTI-HEAD HIERARCHICAL CLASSIFIER")
    print("=" * 80)

    model_multihead = CytologyMultiHead(
        embedding_dim=1536,
        morpho_dim=20,
        hidden_dim=256,
        rejection_threshold=0.5
    )
    print(f"Total Parameters: {count_parameters(model_multihead):,}")
    print(f"Architecture:")
    print(f"  Shared: GFF → 1536 → 512 → 256")
    print(f"  Head 1 (Binary): 256 → 64 → 2")
    print(f"  Head 2 (Severity): 276 → 64 → 2 (includes morpho)")
    print(f"  Head 3 (Fine): 256 → 128 → 6")
    print(f"  Rejection: 259 → 64 → 1")

    model_multihead.train()
    outputs = model_multihead(embedding, morpho, return_all=True)

    print(f"\nOutputs:")
    print(f"  binary_logits: {outputs['binary_logits'].shape}")
    print(f"  severity_logits: {outputs['severity_logits'].shape}")
    print(f"  fine_logits: {outputs['fine_logits'].shape}")
    print(f"  binary_probs: {outputs['binary_probs'].shape}")
    print(f"  severity_probs: {outputs['severity_probs'].shape}")
    print(f"  fine_probs: {outputs['fine_probs'].shape}")
    print(f"  should_reject: {outputs['should_reject'].shape}")
    print(f"  rejection_prob: {outputs['rejection_prob'].shape}")

    # Test hierarchical prediction
    model_multihead.eval()
    clinical_results = model_multihead.get_hierarchical_prediction(embedding, morpho)
    print(f"\nClinical Interpretation (Sample 0):")
    result = clinical_results[0]
    print(f"  Is Normal: {result['is_normal']} (conf: {result['normal_confidence']:.2%})")
    print(f"  Is High-Risk: {result['is_high_risk']} (conf: {result['severity_confidence']:.2%})")
    print(f"  Bethesda Class: {result['bethesda_name']} (conf: {result['bethesda_confidence']:.2%})")
    print(f"  Needs Review: {result['needs_review']} (prob: {result['review_probability']:.2%})")
    print(f"  Priority: {result['priority']}")
    print(f"  Clinical Action: {result['clinical_action']}")

    # ═════════════════════════════════════════════════════════════════════
    #  TEST MULTI-HEAD LOSS
    # ═════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("MULTI-HEAD LOSS TEST")
    print("=" * 80)

    targets_fine = torch.tensor([0, 2, 4, 5])  # NILM, ASCH, HSIL, SCC

    criterion_multihead = MultiHeadLoss(
        lambda_binary=1.0,
        lambda_severity=1.5,
        lambda_fine=1.0,
        use_focal=True
    )

    model_multihead.train()
    outputs = model_multihead(embedding, morpho, return_all=False)
    total_loss, loss_dict = criterion_multihead(outputs, targets_fine)

    print(f"Loss Binary: {loss_dict['loss_binary']:.4f}")
    print(f"Loss Severity: {loss_dict['loss_severity']:.4f}")
    print(f"Loss Fine: {loss_dict['loss_fine']:.4f}")
    print(f"Loss Total: {loss_dict['loss_total']:.4f}")

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
    params_multihead = count_parameters(model_multihead)

    print(f"V14 (Concat):       {params_v14:,} params")
    print(f"V15 (GFF):          {params_v15:,} params")
    print(f"V15.2 (MultiHead):  {params_multihead:,} params")
    print(f"\nOverhead:")
    print(f"  GFF vs Concat:      +{params_v15 - params_v14:,} (+{(params_v15/params_v14 - 1)*100:.1f}%)")
    print(f"  MultiHead vs GFF:   +{params_multihead - params_v15:,} (+{(params_multihead/params_v15 - 1)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("V14, V15 & V15.2 Architectures validated")
    print("Ready for training")
    print("=" * 80)
