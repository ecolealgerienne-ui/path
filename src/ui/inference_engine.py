"""
CellViT-Optimus R&D Cockpit — Moteur d'Inférence Unifié.

Ce module encapsule toute la logique IA pour l'IHM Gradio.
Il réutilise 100% des modules src/ existants sans duplication.

Usage:
    engine = CellVitEngine(device="cuda", family="respiratory")
    result = engine.analyze(image_rgb)
    # result.instance_map, result.metrics, result.report...
"""

import numpy as np
import torch
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

# Modules existants - SINGLE SOURCE OF TRUTH
from src.constants import (
    HOPTIMUS_MEAN,
    HOPTIMUS_STD,
    HOPTIMUS_INPUT_SIZE,
    PANNUKE_IMAGE_SIZE,
    FAMILIES,
)
from src.models.loader import ModelLoader
from src.evaluation.instance_evaluation import run_inference
from src.postprocessing.watershed import hv_guided_watershed
from src.metrics.morphometry import MorphometryAnalyzer, MorphometryReport, CELL_TYPES
from src.preprocessing import preprocess_image, validate_features

logger = logging.getLogger(__name__)


# Paramètres Watershed optimisés par famille (source: CLAUDE.md)
WATERSHED_PARAMS = {
    "respiratory": {"np_threshold": 0.40, "min_size": 30, "beta": 0.50, "min_distance": 5},
    "urologic": {"np_threshold": 0.45, "min_size": 30, "beta": 0.50, "min_distance": 2},
    "epidermal": {"np_threshold": 0.45, "min_size": 20, "beta": 1.00, "min_distance": 3},
    "digestive": {"np_threshold": 0.45, "min_size": 60, "beta": 2.00, "min_distance": 5},
    "glandular": {"np_threshold": 0.40, "min_size": 30, "beta": 0.50, "min_distance": 5},
}

# Checkpoints par famille
CHECKPOINT_PATHS = {
    "respiratory": "models/checkpoints_v13_smart_crops/hovernet_respiratory_v13_smart_crops_hybrid_fpn_best.pth",
    "urologic": "models/checkpoints_v13_smart_crops/hovernet_urologic_v13_smart_crops_hybrid_fpn_best.pth",
    "epidermal": "models/checkpoints_v13_smart_crops/hovernet_epidermal_v13_smart_crops_hybrid_fpn_best.pth",
    "digestive": "models/checkpoints_v13_smart_crops/hovernet_digestive_v13_smart_crops_hybrid_fpn_best.pth",
    "glandular": "models/checkpoints_v13_smart_crops/hovernet_glandular_v13_smart_crops_hybrid_fpn_best.pth",
}

ORGAN_HEAD_PATH = "models/checkpoints/organ_head_best.pth"


@dataclass
class NucleusInfo:
    """Informations sur un noyau individuel."""
    id: int
    centroid: Tuple[int, int]  # (y, x)
    area_um2: float
    perimeter_um: float
    circularity: float
    cell_type: str
    type_idx: int
    confidence: float = 1.0
    is_uncertain: bool = False
    is_mitotic: bool = False


@dataclass
class AnalysisResult:
    """Résultat complet d'une analyse d'image."""

    # Image originale
    image_rgb: np.ndarray

    # Prédictions brutes
    np_pred: np.ndarray  # (H, W) probabilité nucléaire [0, 1]
    hv_pred: np.ndarray  # (2, H, W) gradients HV [-1, 1]
    nt_pred: Optional[np.ndarray] = None  # (5, H, W) types cellulaires

    # Instance segmentation
    instance_map: np.ndarray = None  # (H, W) IDs des instances
    type_map: np.ndarray = None  # (H, W) types par pixel

    # Métriques
    n_nuclei: int = 0
    nucleus_info: List[NucleusInfo] = field(default_factory=list)

    # Morphometry report
    morphometry: Optional[MorphometryReport] = None

    # Incertitude
    uncertainty_map: np.ndarray = None  # (H, W) [0, 1]

    # Organe prédit
    organ_name: str = "Unknown"
    organ_confidence: float = 0.0
    family: str = "unknown"

    # Paramètres utilisés
    watershed_params: Dict = field(default_factory=dict)

    # Debug info
    inference_time_ms: float = 0.0

    def get_nucleus_at(self, y: int, x: int) -> Optional[NucleusInfo]:
        """Retourne les infos du noyau à la position (y, x)."""
        if self.instance_map is None:
            return None
        nucleus_id = self.instance_map[y, x]
        if nucleus_id == 0:
            return None
        for n in self.nucleus_info:
            if n.id == nucleus_id:
                return n
        return None


class CellVitEngine:
    """
    Moteur d'inférence unifié pour CellViT-Optimus.

    Encapsule le chargement des modèles et le pipeline d'analyse complet.
    Les modèles sont chargés UNE SEULE FOIS à l'instanciation.

    Usage:
        engine = CellVitEngine(device="cuda", family="respiratory")
        result = engine.analyze(image_rgb)
    """

    def __init__(
        self,
        device: str = "cuda",
        family: str = "respiratory",
        load_backbone: bool = True,
        load_organ_head: bool = True,
    ):
        """
        Initialise le moteur d'inférence.

        Args:
            device: "cuda" ou "cpu"
            family: Famille d'organes ("respiratory", "urologic", etc.)
            load_backbone: Charger H-optimus-0 (4.5GB, ~5s)
            load_organ_head: Charger OrganHead
        """
        self.device = device
        self.family = family
        self._models_loaded = False

        # Modèles (chargés à la demande)
        self.backbone = None
        self.organ_head = None
        self.hovernet = None

        # Flags modèle (initialisés dans _load_hovernet)
        self._is_hybrid = False
        self._use_fpn_chimique = False
        self._use_h_alpha = False

        # Analyseur morphométrique
        self.morphometry_analyzer = MorphometryAnalyzer(pixel_size_um=0.5)

        # Paramètres watershed pour cette famille
        self.watershed_params = WATERSHED_PARAMS.get(family, WATERSHED_PARAMS["respiratory"])

        # Charger les modèles
        if load_backbone or load_organ_head:
            self._load_models(load_backbone, load_organ_head)

    def _load_models(self, load_backbone: bool, load_organ_head: bool):
        """Charge les modèles depuis les checkpoints."""
        logger.info(f"Loading models for family '{self.family}' on {self.device}...")

        # 1. Backbone H-optimus-0
        if load_backbone:
            logger.info("Loading H-optimus-0 backbone...")
            self.backbone = ModelLoader.load_hoptimus0(device=self.device)
            logger.info("  H-optimus-0 loaded")

        # 2. OrganHead
        if load_organ_head and Path(ORGAN_HEAD_PATH).exists():
            logger.info("Loading OrganHead...")
            self.organ_head = ModelLoader.load_organ_head(
                checkpoint_path=Path(ORGAN_HEAD_PATH),
                device=self.device
            )
            logger.info("  OrganHead loaded")

        # 3. HoVer-Net pour cette famille
        checkpoint_path = CHECKPOINT_PATHS.get(self.family)
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Loading HoVer-Net ({self.family})...")
            self._load_hovernet(checkpoint_path)
            logger.info(f"  HoVer-Net ({self.family}) loaded")
        else:
            logger.warning(f"HoVer-Net checkpoint not found: {checkpoint_path}")

        self._models_loaded = True
        logger.info("All models loaded successfully")

    def _load_hovernet(self, checkpoint_path: str):
        """
        Charge HoVer-Net avec détection automatique du type.

        Lit les flags use_hybrid/use_fpn_chimique/use_h_alpha directement du checkpoint
        (source: scripts/training/train_hovernet_family_v13_smart_crops.py).
        """
        from src.models.hovernet_decoder import HoVerNetDecoder
        from src.models.hovernet_decoder_hybrid import HoVerNetDecoderHybrid

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Lire les flags directement du checkpoint (méthode fiable)
        # Fallback: détection par clés si ancien checkpoint
        use_hybrid = checkpoint.get("use_hybrid", False)
        use_fpn_chimique = checkpoint.get("use_fpn_chimique", False)
        use_h_alpha = checkpoint.get("use_h_alpha", False)

        # Fallback pour anciens checkpoints sans flags explicites
        if not use_hybrid and not use_fpn_chimique:
            # Détection par clés (ancienne méthode)
            use_hybrid = any("h_channel" in k or "fpn" in k.lower() for k in state_dict.keys())
            use_fpn_chimique = use_hybrid  # Si hybrid, probablement FPN chimique

        logger.info(f"  Checkpoint flags: use_hybrid={use_hybrid}, use_fpn_chimique={use_fpn_chimique}, use_h_alpha={use_h_alpha}")

        if use_hybrid:
            self.hovernet = HoVerNetDecoderHybrid(
                embed_dim=1536,
                n_classes=5,
                use_fpn_chimique=use_fpn_chimique,
                use_h_alpha=use_h_alpha
            )
        else:
            self.hovernet = HoVerNetDecoder(
                embed_dim=1536,
                n_classes=5,
            )

        # Nettoyer les clés (module. prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "").replace("model.", "")
            new_state_dict[name] = v

        self.hovernet.load_state_dict(new_state_dict, strict=False)
        self.hovernet = self.hovernet.to(self.device)
        self.hovernet.eval()

        # Stocker les flags pour l'inférence
        self._is_hybrid = use_hybrid
        self._use_fpn_chimique = use_fpn_chimique
        self._use_h_alpha = use_h_alpha

    def _preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        """
        Prétraite une image pour l'inférence.

        Utilise preprocess_image() de src.preprocessing (source unique de vérité).

        Args:
            image: Image RGB (H, W, 3) uint8, DOIT être 224×224

        Returns:
            Tuple (tensor_normalized, image_224, images_rgb):
                - tensor_normalized: Tensor (1,3,224,224) normalisé pour H-optimus-0
                - image_224: Image numpy (224,224,3) uint8
                - images_rgb: Tensor (1,3,224,224) [0,1] pour FPN Chimique
        """
        import torchvision.transforms as T

        # Validation: L'image DOIT être 224×224 (validation en amont dans app.py)
        if image.shape[0] != HOPTIMUS_INPUT_SIZE or image.shape[1] != HOPTIMUS_INPUT_SIZE:
            raise ValueError(
                f"Image must be {HOPTIMUS_INPUT_SIZE}×{HOPTIMUS_INPUT_SIZE}, "
                f"got {image.shape[0]}×{image.shape[1]}"
            )

        # Copie de l'image (déjà 224×224)
        image_224 = image.copy()

        # Conversion uint8 si nécessaire
        if image_224.dtype != np.uint8:
            if image_224.max() <= 1.0:
                image_224 = (image_224 * 255).clip(0, 255).astype(np.uint8)
            else:
                image_224 = image_224.clip(0, 255).astype(np.uint8)

        # Tensor normalisé pour H-optimus-0 (méthode centralisée)
        tensor_normalized = preprocess_image(image_224, device=self.device)

        # Tensor [0,1] pour FPN Chimique (images_rgb)
        # CRITIQUE: Le DataLoader entraînement utilise ToTensor() → [0,1]
        images_rgb = T.ToTensor()(image_224).unsqueeze(0).to(self.device)

        return tensor_normalized, image_224, images_rgb

    def analyze(
        self,
        image: np.ndarray,
        watershed_params: Optional[Dict] = None,
        compute_morphometry: bool = True,
        compute_uncertainty: bool = True,
    ) -> AnalysisResult:
        """
        Analyse complète d'une image H&E.

        Args:
            image: Image RGB (H, W, 3) uint8, DOIT être 224×224 (taille H-optimus-0)
            watershed_params: Override des paramètres watershed
            compute_morphometry: Calculer les métriques morphométriques
            compute_uncertainty: Calculer la carte d'incertitude

        Returns:
            AnalysisResult avec toutes les prédictions et métriques
        """
        import time
        start_time = time.time()

        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call _load_models() first.")

        # Paramètres watershed
        params = {**self.watershed_params, **(watershed_params or {})}

        # Prétraitement (méthode centralisée)
        tensor, image_224, images_rgb = self._preprocess_image(image)

        # Extraction features
        with torch.no_grad():
            features = self.backbone.forward_features(tensor)  # (1, 261, 1536)

        # Validation des features (détection bugs preprocessing)
        validation = validate_features(features)
        if not validation["valid"]:
            logger.warning(f"Feature validation warning: {validation['message']}")
        else:
            logger.debug(f"Features OK: CLS std={validation['cls_std']:.3f}")

        # Prédiction organe (optionnel)
        organ_name = "Unknown"
        organ_confidence = 0.0
        if self.organ_head is not None:
            cls_token = features[:, 0, :]  # (1, 1536)
            organ_logits = self.organ_head(cls_token)
            organ_probs = torch.softmax(organ_logits, dim=1)
            organ_idx = organ_probs.argmax(dim=1).item()
            organ_confidence = organ_probs[0, organ_idx].item()

            # Mapper l'index vers le nom (source: scripts/training/train_organ_head.py)
            ORGAN_NAMES = [
                "Adrenal_gland", "Bile-duct", "Bladder", "Breast", "Cervix",
                "Colon", "Esophagus", "HeadNeck", "Kidney", "Liver",
                "Lung", "Ovarian", "Pancreatic", "Prostate", "Skin",
                "Stomach", "Testis", "Thyroid", "Uterus"
            ]
            if organ_idx < len(ORGAN_NAMES):
                organ_name = ORGAN_NAMES[organ_idx]

        # Inférence HoVer-Net (utilise run_inference de src.evaluation)
        # images_rgb passé uniquement si modèle hybrid (FPN Chimique)
        np_pred, hv_pred = run_inference(
            model=self.hovernet,
            features=features,
            images_rgb=images_rgb if self._is_hybrid else None,
            device=self.device
        )

        # Watershed
        instance_map = hv_guided_watershed(
            np_pred, hv_pred,
            np_threshold=params["np_threshold"],
            beta=params["beta"],
            min_size=params["min_size"],
            min_distance=params["min_distance"]
        )

        # Nombre d'instances
        n_nuclei = len(np.unique(instance_map)) - 1  # Exclure background

        # Type map (argmax de NT si disponible, sinon 0)
        type_map = np.zeros_like(instance_map, dtype=np.int32)

        # Incertitude
        uncertainty_map = None
        if compute_uncertainty:
            uncertainty_map = self._compute_uncertainty(np_pred, hv_pred)

        # Morphométrie
        morphometry = None
        nucleus_info = []
        if compute_morphometry and n_nuclei > 0:
            morphometry = self.morphometry_analyzer.analyze(
                instance_map, type_map, image=image_224
            )

            # Extraire infos par noyau
            nucleus_info = self._extract_nucleus_info(
                instance_map, type_map, np_pred, uncertainty_map
            )

        inference_time = (time.time() - start_time) * 1000

        return AnalysisResult(
            image_rgb=image_224,
            np_pred=np_pred,
            hv_pred=hv_pred,
            instance_map=instance_map,
            type_map=type_map,
            n_nuclei=n_nuclei,
            nucleus_info=nucleus_info,
            morphometry=morphometry,
            uncertainty_map=uncertainty_map,
            organ_name=organ_name,
            organ_confidence=organ_confidence,
            family=self.family,
            watershed_params=params,
            inference_time_ms=inference_time,
        )

    def _compute_uncertainty(
        self,
        np_pred: np.ndarray,
        hv_pred: np.ndarray
    ) -> np.ndarray:
        """
        Calcule une carte d'incertitude.

        Sources:
        - NP proche de 0.5 = incertain
        - HV magnitude faible = incertain
        """
        # NP proche de 0.5 = maximum d'incertitude
        np_uncertainty = 1 - np.abs(np_pred - 0.5) * 2  # [0, 1]

        # HV magnitude faible = incertain sur la direction
        hv_magnitude = np.sqrt(hv_pred[0]**2 + hv_pred[1]**2)
        hv_max = np.sqrt(2)  # Max théorique
        hv_uncertainty = 1 - (hv_magnitude / hv_max)  # [0, 1]

        # Combinaison pondérée
        uncertainty = 0.6 * np_uncertainty + 0.4 * hv_uncertainty

        return uncertainty.astype(np.float32)

    def _extract_nucleus_info(
        self,
        instance_map: np.ndarray,
        type_map: np.ndarray,
        np_pred: np.ndarray,
        uncertainty_map: Optional[np.ndarray]
    ) -> List[NucleusInfo]:
        """Extrait les informations détaillées par noyau."""
        nuclei = []

        for inst_id in np.unique(instance_map):
            if inst_id == 0:
                continue

            mask = instance_map == inst_id
            area_pixels = mask.sum()

            if area_pixels < 10:
                continue

            # Centroïde
            coords = np.where(mask)
            cy, cx = int(coords[0].mean()), int(coords[1].mean())

            # Périmètre et circularité
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            perimeter = 0.0
            circularity = 0.0
            if len(contours) > 0:
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    circularity = (4 * np.pi * area_pixels) / (perimeter ** 2)
                    circularity = min(circularity, 1.0)

            # Type cellulaire
            types_in_mask = type_map[mask]
            type_idx = int(np.bincount(types_in_mask.astype(int)).argmax()) if len(types_in_mask) > 0 else 0
            cell_type = CELL_TYPES[type_idx] if type_idx < len(CELL_TYPES) else "Unknown"

            # Confiance (moyenne NP dans le masque)
            confidence = np_pred[mask].mean()

            # Incertitude
            is_uncertain = False
            if uncertainty_map is not None:
                mean_uncertainty = uncertainty_map[mask].mean()
                is_uncertain = mean_uncertainty > 0.5

            nuclei.append(NucleusInfo(
                id=inst_id,
                centroid=(cy, cx),
                area_um2=area_pixels * 0.25,  # 0.5^2 MPP
                perimeter_um=perimeter * 0.5,
                circularity=circularity,
                cell_type=cell_type,
                type_idx=type_idx,
                confidence=confidence,
                is_uncertain=is_uncertain,
            ))

        return nuclei

    def change_family(self, family: str):
        """Change la famille et recharge HoVer-Net correspondant."""
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}. Valid: {FAMILIES}")

        self.family = family
        self.watershed_params = WATERSHED_PARAMS.get(family, WATERSHED_PARAMS["respiratory"])

        checkpoint_path = CHECKPOINT_PATHS.get(family)
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Switching to HoVer-Net ({family})...")
            self._load_hovernet(checkpoint_path)
        else:
            logger.warning(f"HoVer-Net checkpoint not found for {family}")

    @property
    def is_ready(self) -> bool:
        """Vérifie si le moteur est prêt pour l'inférence."""
        return (
            self._models_loaded
            and self.backbone is not None
            and self.hovernet is not None
        )

    def get_status(self) -> Dict:
        """Retourne le status du moteur."""
        return {
            "models_loaded": self._models_loaded,
            "backbone_loaded": self.backbone is not None,
            "organ_head_loaded": self.organ_head is not None,
            "hovernet_loaded": self.hovernet is not None,
            "family": self.family,
            "device": self.device,
            "watershed_params": self.watershed_params,
            "is_hybrid": self._is_hybrid,
            "use_fpn_chimique": self._use_fpn_chimique,
            "use_h_alpha": self._use_h_alpha,
        }
