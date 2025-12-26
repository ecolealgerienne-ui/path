"""
Interface standardisée pour tous les modèles du projet.

Ce module définit des wrappers qui normalisent les entrées/sorties,
évitant les erreurs de compatibilité lors de changements d'implémentation.

Principe: Les scripts d'évaluation/inférence ne doivent JAMAIS dépendre
de la structure interne des modèles (tuple, dict, etc.).
"""

from typing import Dict, Any, Tuple, Optional
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class HoVerNetOutput:
    """
    Format standardisé pour les sorties HoVer-Net.

    Tous les wrappers HoVer-Net DOIVENT retourner ce format,
    quelle que soit l'implémentation interne (tuple, dict, etc.).

    NOTE IMPORTANTE (2025-12-24):
    - NP produit 2 canaux (background/foreground) avec CrossEntropyLoss
    - Pour obtenir le masque binaire, utiliser softmax puis canal 1 (foreground)
    - NE PAS utiliser sigmoid sur 2 canaux!
    """
    np: torch.Tensor  # (B, 2, H, W) - Nuclear Presence logits (bg/fg)
    hv: torch.Tensor  # (B, 2, H, W) - Horizontal/Vertical gradients
    nt: torch.Tensor  # (B, n_classes, H, W) - Nuclear Type logits

    def to_numpy(self, apply_activations: bool = True) -> Dict[str, np.ndarray]:
        """
        Convertit en numpy avec activations optionnelles.

        Args:
            apply_activations: Si True, applique softmax(NP) et softmax(NT)

        Returns:
            {
                "np": (H, W) float32 dans [0, 1] si activations=True (foreground prob)
                "hv": (2, H, W) float32 dans [-1, 1]
                "nt": (n_classes, H, W) float32 dans [0, 1] si activations=True
            }
        """
        result = {}

        # NP: (B, 2, H, W) → (H, W)
        # IMPORTANT: Prendre canal 1 (foreground) après softmax
        np_out = self.np[0]  # (2, H, W)
        if apply_activations:
            np_out = torch.softmax(np_out, dim=0)[1]  # Canal 1 = foreground
        else:
            np_out = np_out[1]  # Logits foreground
        result["np"] = np_out.cpu().numpy()

        # HV: (B, 2, H, W) → (2, H, W)
        result["hv"] = self.hv[0].cpu().numpy()

        # NT: (B, n_classes, H, W) → (n_classes, H, W)
        nt_out = self.nt[0]
        if apply_activations:
            nt_out = torch.softmax(nt_out, dim=0)
        result["nt"] = nt_out.cpu().numpy()

        return result


class HoVerNetWrapper:
    """
    Wrapper standardisé pour HoVer-Net.

    Masque les détails d'implémentation (tuple vs dict) et garantit
    une interface stable pour tous les scripts d'évaluation.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        """
        Args:
            model: HoVerNetDecoder ou tout modèle compatible
            device: Device PyTorch
        """
        self.model = model
        self.device = device
        self.model.eval()

    def __call__(self, features: torch.Tensor) -> HoVerNetOutput:
        """
        Inférence avec sortie standardisée.

        Args:
            features: (B, N, D) - Patch tokens ou CLS+patches

        Returns:
            HoVerNetOutput avec format uniforme
        """
        with torch.no_grad():
            # L'implémentation interne peut changer (tuple, dict, objet, etc.)
            # Ce wrapper gère tous les cas
            raw_output = self.model(features)

            # Détection automatique du format
            if isinstance(raw_output, tuple):
                # Format tuple: (np_out, hv_out, nt_out)
                np_out, hv_out, nt_out = raw_output
            elif isinstance(raw_output, dict):
                # Format dict: {"np": ..., "hv": ..., "nt": ...}
                np_out = raw_output["np"]
                hv_out = raw_output["hv"]
                nt_out = raw_output["nt"]
            elif hasattr(raw_output, 'np') and hasattr(raw_output, 'hv'):
                # Format dataclass/objet
                np_out = raw_output.np
                hv_out = raw_output.hv
                nt_out = raw_output.nt
            else:
                raise TypeError(
                    f"Format de sortie HoVer-Net non supporté: {type(raw_output)}. "
                    f"Attendu: tuple, dict ou dataclass avec attributs np/hv/nt."
                )

            return HoVerNetOutput(np=np_out, hv=hv_out, nt=nt_out)


@dataclass
class OrganHeadOutput:
    """
    Format standardisé pour les sorties OrganHead.
    """
    logits: torch.Tensor  # (B, n_organs) - Logits bruts
    organ_idx: int  # Index de l'organe prédit
    organ_name: str  # Nom de l'organe (ex: "Breast")
    confidence: float  # Confiance (softmax max)
    probabilities: torch.Tensor  # (B, n_organs) - Probabilités calibrées

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire sérialisable."""
        return {
            "organ_idx": self.organ_idx,
            "organ_name": self.organ_name,
            "confidence": float(self.confidence),
            "probabilities": self.probabilities.cpu().numpy().tolist(),
        }


class OrganHeadWrapper:
    """
    Wrapper standardisé pour OrganHead.

    Gère la normalisation des sorties et la conversion organe_idx → nom.
    """

    # Mapping standard (doit correspondre à PANNUKE_ORGANS)
    ORGAN_NAMES = [
        "Adrenal_gland", "Bile-duct", "Bladder", "Breast", "Cervix",
        "Colon", "Esophagus", "HeadNeck", "Kidney", "Liver",
        "Lung", "Ovarian", "Pancreatic", "Prostate", "Skin",
        "Stomach", "Testis", "Thyroid", "Uterus"
    ]

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        temperature: float = 0.5
    ):
        """
        Args:
            model: OrganHead model
            device: Device PyTorch
            temperature: Temperature Scaling (défaut: 0.5 pour calibration)
        """
        self.model = model
        self.device = device
        self.temperature = temperature
        self.model.eval()

    def __call__(self, features: torch.Tensor) -> OrganHeadOutput:
        """
        Inférence avec sortie standardisée.

        Args:
            features: (B, D) - CLS token typiquement

        Returns:
            OrganHeadOutput avec format uniforme
        """
        with torch.no_grad():
            logits = self.model(features)  # (B, n_organs)

            # Temperature scaling
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=1)

            # Prédiction
            max_prob, organ_idx = probs.max(dim=1)
            organ_idx = organ_idx.item()
            organ_name = self.ORGAN_NAMES[organ_idx]
            confidence = max_prob.item()

            return OrganHeadOutput(
                logits=logits,
                organ_idx=organ_idx,
                organ_name=organ_name,
                confidence=confidence,
                probabilities=probs
            )


class BackboneWrapper:
    """
    Wrapper standardisé pour H-optimus-0 (ou tout backbone ViT).

    Garantit que les features extraites sont toujours au bon format,
    avec validation automatique.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        """
        Args:
            model: H-optimus-0 ou backbone compatible
            device: Device PyTorch
        """
        self.model = model
        self.device = device
        self.model.eval()

    def __call__(self, tensor: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """
        Extraction de features avec validation optionnelle.

        Args:
            tensor: (B, 3, 224, 224) - Image prétraitée
            validate: Si True, vérifie CLS std dans [0.70-0.90]

        Returns:
            features: (B, 261, 1536) - CLS token + 256 patches

        Raises:
            ValueError: Si validation échoue (CLS std hors range)
        """
        with torch.no_grad():
            # TOUJOURS utiliser forward_features() (inclut LayerNorm)
            features = self.model.forward_features(tensor)

        if validate:
            from src.preprocessing import validate_features
            validation = validate_features(features)

            if not validation["valid"]:
                raise ValueError(
                    f"Features invalides: {validation['message']}. "
                    f"CLS std: {validation.get('cls_std', 'N/A')} "
                    f"(attendu: 0.70-0.90)"
                )

        return features.float()

    def extract_cls_token(self, features: torch.Tensor) -> torch.Tensor:
        """Extrait le CLS token (première position)."""
        return features[:, 0, :]

    def extract_patch_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extrait les patch tokens spatiaux (positions 5-260).

        Structure H-optimus-0:
        - Index 0: CLS token
        - Index 1-4: Register tokens (non-spatiaux)
        - Index 5-260: Patch tokens (grille 16x16 spatiale)

        FIX Register Token (2025-12-25): 5:261 au lieu de 1:257
        """
        return features[:, 5:261, :]


def create_hovernet_wrapper(
    checkpoint_path: str,
    device: str = "cuda"
) -> HoVerNetWrapper:
    """
    Factory pour créer un HoVerNetWrapper depuis un checkpoint.

    Utilise ModelLoader en interne mais retourne un wrapper standardisé.

    Args:
        checkpoint_path: Chemin vers le checkpoint
        device: Device PyTorch

    Returns:
        HoVerNetWrapper prêt à l'emploi
    """
    from src.models.loader import ModelLoader
    from pathlib import Path

    model = ModelLoader.load_hovernet(
        checkpoint_path=Path(checkpoint_path),
        device=device
    )

    return HoVerNetWrapper(model, device)


def create_organ_head_wrapper(
    checkpoint_path: str,
    device: str = "cuda",
    temperature: float = 0.5
) -> OrganHeadWrapper:
    """
    Factory pour créer un OrganHeadWrapper depuis un checkpoint.

    Args:
        checkpoint_path: Chemin vers le checkpoint
        device: Device PyTorch
        temperature: Temperature Scaling

    Returns:
        OrganHeadWrapper prêt à l'emploi
    """
    from src.models.loader import ModelLoader
    from pathlib import Path

    model = ModelLoader.load_organ_head(
        checkpoint_path=Path(checkpoint_path),
        device=device
    )

    return OrganHeadWrapper(model, device, temperature)


def create_backbone_wrapper(device: str = "cuda") -> BackboneWrapper:
    """
    Factory pour créer un BackboneWrapper (H-optimus-0).

    Args:
        device: Device PyTorch

    Returns:
        BackboneWrapper prêt à l'emploi
    """
    from src.models.loader import ModelLoader

    model = ModelLoader.load_hoptimus0(device=device)

    return BackboneWrapper(model, device)
