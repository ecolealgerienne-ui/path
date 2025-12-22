"""
Module centralisé pour chargement des modèles.

CE MODULE EST LA SOURCE UNIQUE pour charger tous les modèles du projet:
- H-optimus-0 (backbone foundation model)
- OrganHead (classification organe)
- HoVer-Net (segmentation cellulaire par famille)

Avantages:
- Gestion d'erreur unifiée
- Configuration cohérente
- Pas de duplication de logique

Usage:
    >>> from src.models.loader import ModelLoader
    >>> backbone = ModelLoader.load_hoptimus0(device="cuda")
    >>> organ_head = ModelLoader.load_organ_head(checkpoint_path, device="cuda")
    >>> hovernet = ModelLoader.load_hovernet(checkpoint_path, device="cuda")
"""

import timm
import torch
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

__all__ = ['ModelLoader']


class ModelLoader:
    """
    Chargeur unifié pour tous les modèles du projet.

    Cette classe centralise la logique de chargement pour éviter
    les duplications et garantir la cohérence.

    Methods:
        load_hoptimus0: Charge le backbone H-optimus-0
        load_organ_head: Charge OrganHead depuis checkpoint
        load_hovernet: Charge HoVer-Net depuis checkpoint
    """

    @staticmethod
    def load_hoptimus0(
        device: str = "cuda",
        cache_dir: Optional[Path] = None,
        force_download: bool = False
    ) -> torch.nn.Module:
        """
        Charge H-optimus-0 depuis HuggingFace.

        H-optimus-0 est un Vision Transformer pré-entraîné par Bioptimus
        sur 500k+ lames H&E. Il sert de backbone pour l'extraction de features.

        Architecture:
            - ViT-Giant/14 avec 4 registres
            - 1.1 milliard de paramètres
            - Entrée: 224×224 pixels
            - Sortie: 261 tokens × 1536 dim (1 CLS + 256 patches + 4 registers)

        IMPORTANT:
            - Le modèle est TOUJOURS gelé (requires_grad=False)
            - Accès gated sur HuggingFace (token requis)
            - Utiliser UNIQUEMENT forward_features() pour extraction

        Args:
            device: Device PyTorch ("cuda", "cpu", "mps")
            cache_dir: Répertoire cache HuggingFace (optionnel)
            force_download: Force le téléchargement même si déjà en cache

        Returns:
            Modèle H-optimus-0 gelé en mode eval

        Raises:
            RuntimeError: Si accès refusé (token HuggingFace invalide)
            ValueError: Si device invalide

        Example:
            >>> backbone = ModelLoader.load_hoptimus0(device="cuda")
            >>> features = backbone.forward_features(tensor)
            >>> features.shape
            torch.Size([1, 261, 1536])

        Notes:
            - Première exécution: télécharge ~4.5 GB
            - Ensuite: chargement depuis cache (~5 secondes)
            - Token HF requis: huggingface-cli login

        Accès HuggingFace:
            1. Créer compte sur https://huggingface.co
            2. Demander accès: https://huggingface.co/bioptimus/H-optimus-0
            3. Créer token avec "Read access to public gated repos"
            4. Login: huggingface-cli login

        See Also:
            - https://huggingface.co/bioptimus/H-optimus-0
            - CLAUDE.md Section "Accès H-optimus-0"
        """
        logger.info("Loading H-optimus-0 backbone...")

        # Validation device
        if device not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Invalid device: {device}. Must be 'cpu', 'cuda', or 'mps'")

        try:
            # Charger depuis HuggingFace via timm
            model = timm.create_model(
                "hf-hub:bioptimus/H-optimus-0",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False
            )

            # Transfert vers device
            model = model.to(device)

            # Mode eval (désactive dropout, batchnorm en mode inference)
            model.eval()

            # Geler TOUS les paramètres (backbone ne s'entraîne JAMAIS)
            for param in model.parameters():
                param.requires_grad = False

            # Compter paramètres
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"✅ H-optimus-0 loaded: {total_params:,} parameters on {device}")

            return model

        except Exception as e:
            error_msg = str(e).lower()

            # Erreur d'accès (401/403)
            if "401" in error_msg or "403" in error_msg or "unauthorized" in error_msg:
                raise RuntimeError(
                    "❌ Accès H-optimus-0 refusé (401/403)\n\n"
                    "SOLUTION:\n"
                    "1. Créer compte HuggingFace: https://huggingface.co\n"
                    "2. Demander accès: https://huggingface.co/bioptimus/H-optimus-0\n"
                    "3. Créer token avec 'Read access to public gated repos'\n"
                    "4. Login: huggingface-cli login\n\n"
                    f"Erreur originale: {e}"
                )

            # Erreur de connexion
            elif "connection" in error_msg or "network" in error_msg:
                raise RuntimeError(
                    "❌ Erreur de connexion HuggingFace\n\n"
                    "SOLUTION:\n"
                    "1. Vérifier connexion internet\n"
                    "2. Vérifier proxy/firewall\n"
                    "3. Réessayer dans quelques minutes\n\n"
                    f"Erreur originale: {e}"
                )

            # Autre erreur
            else:
                raise RuntimeError(
                    f"❌ Erreur chargement H-optimus-0: {e}\n\n"
                    f"Voir logs pour plus de détails."
                )

    @staticmethod
    def load_organ_head(
        checkpoint_path: Path,
        device: str = "cuda",
        num_organs: int = 19,
        embed_dim: int = 1536
    ) -> torch.nn.Module:
        """
        Charge OrganHead depuis checkpoint.

        OrganHead est un MLP qui prédit l'organe d'origine à partir
        du CLS token de H-optimus-0.

        Architecture:
            - Input: CLS token (1536-dim)
            - Hidden: 512 → 256
            - Output: 19 organes (PanNuke)
            - Activation: ReLU + Dropout(0.1)

        Performance:
            - Accuracy: 99.94% (validation 3 folds)
            - Organes à 100%: 15/19

        Args:
            checkpoint_path: Chemin vers le fichier .pth
            device: Device PyTorch
            num_organs: Nombre d'organes (19 pour PanNuke)
            embed_dim: Dimension du CLS token (1536 pour H-optimus-0)

        Returns:
            Modèle OrganHead en mode eval

        Raises:
            FileNotFoundError: Si checkpoint n'existe pas
            RuntimeError: Si chargement échoue

        Example:
            >>> organ_head = ModelLoader.load_organ_head(
            ...     Path("models/checkpoints/organ_head_best.pth"),
            ...     device="cuda"
            ... )
            >>> probs = organ_head(cls_token)
            >>> organ_id = probs.argmax(dim=1).item()

        See Also:
            - src/models/organ_head.py: Architecture détaillée
        """
        from src.models.organ_head import OrganHead

        logger.info(f"Loading OrganHead from {checkpoint_path}...")

        # Validation fichier
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"❌ Checkpoint OrganHead introuvable: {checkpoint_path}\n\n"
                f"Vérifier le chemin ou entraîner le modèle:\n"
                f"  python scripts/training/train_organ_head.py"
            )

        try:
            # Créer modèle
            model = OrganHead(embed_dim=embed_dim, num_organs=num_organs)

            # Charger poids
            state_dict = torch.load(checkpoint_path, map_location=device)

            # Gérer différents formats de checkpoint
            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            else:
                model.load_state_dict(state_dict)

            # Transfert + eval
            model = model.to(device)
            model.eval()

            logger.info(f"✅ OrganHead loaded from {checkpoint_path.name}")

            return model

        except Exception as e:
            raise RuntimeError(
                f"❌ Erreur chargement OrganHead: {e}\n\n"
                f"Checkpoint: {checkpoint_path}\n"
                f"Vérifier compatibilité de version."
            )

    @staticmethod
    def load_hovernet(
        checkpoint_path: Path,
        device: str = "cuda",
        embed_dim: int = 1536,
        num_classes: int = 6,
        dropout: float = 0.1
    ) -> torch.nn.Module:
        """
        Charge HoVer-Net depuis checkpoint.

        HoVer-Net est un décodeur qui segmente et classifie les cellules
        à partir des patch tokens de H-optimus-0.

        Architecture:
            - Input: 256 patch tokens (16×16 grid) × 1536-dim
            - 3 branches: NP (segmentation), HV (gradients), NT (classification)
            - Output: Cartes 256×256 pour NP/HV/NT

        Performance (famille Glandulaire):
            - NP Dice: 0.9648
            - HV MSE: 0.0106
            - NT Accuracy: 0.9111

        Args:
            checkpoint_path: Chemin vers le fichier .pth
            device: Device PyTorch
            embed_dim: Dimension des patch tokens (1536)
            num_classes: Nombre de classes cellulaires (6 = BG + 5 types)
            dropout: Dropout rate (0.1)

        Returns:
            Modèle HoVer-Net en mode eval

        Raises:
            FileNotFoundError: Si checkpoint n'existe pas
            RuntimeError: Si chargement échoue

        Example:
            >>> hovernet = ModelLoader.load_hovernet(
            ...     Path("models/checkpoints/hovernet_glandular_best.pth"),
            ...     device="cuda"
            ... )
            >>> outputs = hovernet(patch_tokens)
            >>> np_mask = outputs["np"].argmax(dim=1)  # (B, 256, 256)

        See Also:
            - src/models/hovernet_decoder.py: Architecture détaillée
        """
        from src.models.hovernet_decoder import HoVerNetDecoder

        logger.info(f"Loading HoVer-Net from {checkpoint_path}...")

        # Validation fichier
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"❌ Checkpoint HoVer-Net introuvable: {checkpoint_path}\n\n"
                f"Vérifier le chemin ou entraîner le modèle:\n"
                f"  python scripts/training/train_hovernet_family.py --family glandular"
            )

        try:
            # Charger checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Créer modèle
            model = HoVerNetDecoder(
                embed_dim=embed_dim,
                num_classes=num_classes,
                dropout=dropout
            )

            # Charger poids
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            # Transfert + eval
            model = model.to(device)
            model.eval()

            # Log métriques si disponibles
            if "metrics" in checkpoint:
                metrics = checkpoint["metrics"]
                logger.info(
                    f"✅ HoVer-Net loaded: "
                    f"Dice={metrics.get('dice', 'N/A'):.4f}, "
                    f"HV MSE={metrics.get('hv_mse', 'N/A'):.4f}"
                )
            else:
                logger.info(f"✅ HoVer-Net loaded from {checkpoint_path.name}")

            return model

        except Exception as e:
            raise RuntimeError(
                f"❌ Erreur chargement HoVer-Net: {e}\n\n"
                f"Checkpoint: {checkpoint_path}\n"
                f"Vérifier compatibilité de version."
            )

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> Dict[str, any]:
        """
        Retourne les informations sur un modèle.

        Utile pour debugging et logging.

        Args:
            model: Modèle PyTorch

        Returns:
            dict avec infos (params, device, trainable)
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        device = next(model.parameters()).device if total_params > 0 else "unknown"

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": total_params - trainable_params,
            "device": str(device),
            "is_trainable": trainable_params > 0,
        }
