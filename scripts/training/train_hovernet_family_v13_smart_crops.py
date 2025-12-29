#!/usr/bin/env python3
"""
Entraînement du décodeur HoVer-Net avec stratégie V13 Smart Crops.

Cette version utilise les données avec splits train/val pré-séparés pour prévenir
le data leakage (stratégie split-first-then-rotate validée par le CTO).

IMPORTANT: Exécuter d'abord prepare_v13_smart_crops.py et extract_features_v13_smart_crops.py:
    1. python scripts/preprocessing/prepare_v13_smart_crops.py --family epidermal
    2. python scripts/preprocessing/extract_features_v13_smart_crops.py --family epidermal --split train
    3. python scripts/preprocessing/extract_features_v13_smart_crops.py --family epidermal --split val

Usage:
    python scripts/training/train_hovernet_family_v13_smart_crops.py \
        --family epidermal \
        --epochs 30 \
        --batch_size 16

Différences avec train_hovernet_family.py:
    - Charge train/val séparés (pas de split automatique)
    - Utilise fichiers *_v13_smart_crops_*.npz
    - Data leakage prevention garanti
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import modules du projet
from src.data.preprocessing import load_targets, resize_targets
from src.models.hovernet_decoder import HoVerNetDecoder, HoVerNetLoss
from src.models.organ_families import FAMILIES, get_organs, FAMILY_DESCRIPTIONS


class FeatureAugmentation:
    """
    Augmentation ALIGNÉE pour features H-optimus-0, targets, weight_map ET image RGB.

    CRITIQUE (2025-12-28): L'image RGB DOIT être augmentée avec la MÊME transformation
    que les features, sinon le H-channel extrait sera désaligné avec les targets.

    Bug corrigé: Avant, l'image n'était pas augmentée → signaux contradictoires
    → AJI 0.4584 (catastrophique) au lieu de 0.5444 (sans augmentation).
    """

    def __init__(self, p_flip: float = 0.5, p_rot90: float = 0.5):
        self.p_flip = p_flip
        self.p_rot90 = p_rot90

    def __call__(self, features, np_target, hv_target, nt_target, weight_map=None, image=None):
        """
        Applique la MÊME transformation géométrique à tous les inputs.

        Args:
            features: (261, 1536) H-optimus-0 features
            np_target: (224, 224) NP target
            hv_target: (2, 224, 224) HV target
            nt_target: (224, 224) NT target
            weight_map: (224, 224) optional Ronneberger weights
            image: (224, 224, 3) optional RGB image for hybrid mode

        Returns:
            Tuple of augmented tensors (MÊME transformation appliquée à tous)
        """
        # Structure H-optimus-0: [CLS, Registers(4), Patches(256)]
        cls_token = features[0:1]
        registers = features[1:5]
        patches = features[5:261]

        # Reshape patches en grille 16x16
        patches_grid = patches.reshape(16, 16, -1)

        # Décision d'augmentation (stockées pour appliquer la MÊME aux deux)
        do_flip = np.random.random() < self.p_flip
        do_rot = np.random.random() < self.p_rot90
        rot_k = np.random.choice([1, 2, 3]) if do_rot else 0

        # Flip horizontal - MÊME transformation pour features, targets ET image
        if do_flip:
            patches_grid = np.flip(patches_grid, axis=1).copy()
            np_target = np.flip(np_target, axis=1).copy()
            hv_target = np.flip(hv_target, axis=2).copy()
            hv_target[0] = -hv_target[0]  # Inverser composante H (index 0)
            nt_target = np.flip(nt_target, axis=1).copy()
            if weight_map is not None:
                weight_map = np.flip(weight_map, axis=1).copy()
            if image is not None:
                image = np.flip(image, axis=1).copy()  # (H, W, C) → flip sur W

        # Rotation 90° - MÊME transformation pour features, targets ET image
        if do_rot and rot_k > 0:
            patches_grid = np.rot90(patches_grid, rot_k, axes=(0, 1)).copy()
            np_target = np.rot90(np_target, rot_k).copy()
            hv_target = np.rot90(hv_target, rot_k, axes=(1, 2)).copy()
            nt_target = np.rot90(nt_target, rot_k).copy()
            if weight_map is not None:
                weight_map = np.rot90(weight_map, rot_k).copy()
            if image is not None:
                image = np.rot90(image, rot_k, axes=(0, 1)).copy()  # (H, W, C) → rot sur H,W

            # Component swapping selon rotation
            if rot_k == 1:  # 90° anti-horaire
                hv_target = np.stack([hv_target[1], -hv_target[0]])
            elif rot_k == 2:  # 180°
                hv_target = np.stack([-hv_target[0], -hv_target[1]])
            elif rot_k == 3:  # 270°
                hv_target = np.stack([-hv_target[1], hv_target[0]])

        patches = patches_grid.reshape(256, -1)
        features = np.concatenate([cls_token, registers, patches], axis=0)

        return features, np_target, hv_target, nt_target, weight_map, image


class V13SmartCropsDataset(Dataset):
    """
    Dataset V13 Smart Crops avec split explicite (train ou val).

    Différence clé avec FamilyHoVerDataset:
    - Pas de split automatique 80/20
    - Charge fichiers train ou val explicitement
    - Data leakage prevention garanti
    - Support mode hybride (charge images RGB pour injection H-channel)
    """

    def __init__(self, family: str, split: str, cache_dir: str = None, augment: bool = False, use_hybrid: bool = False):
        assert split in ["train", "val"], f"Split doit être 'train' ou 'val', pas '{split}'"

        self.family = family
        self.split = split
        self.augment = augment
        self.use_hybrid = use_hybrid
        self.augmenter = FeatureAugmentation() if augment else None

        # Répertoire cache
        if cache_dir is None:
            cache_dir = PROJECT_ROOT / "data/cache/family_data"
        else:
            cache_dir = Path(cache_dir)

        # Chemins fichiers V13 Smart Crops
        features_path = cache_dir / f"{family}_rgb_features_v13_smart_crops_{split}.npz"

        # Pour les targets, on utilise le fichier de prepare_v13_smart_crops.py
        targets_dir = PROJECT_ROOT / "data/family_data_v13_smart_crops"
        targets_path = targets_dir / f"{family}_{split}_v13_smart_crops.npz"

        if not features_path.exists():
            raise FileNotFoundError(
                f"Features V13 Smart Crops ({split}) non trouvées.\n"
                f"Fichier manquant: {features_path}\n\n"
                f"Lancez d'abord:\n"
                f"  python scripts/preprocessing/extract_features_v13_smart_crops.py \\\n"
                f"      --family {family} --split {split}"
            )

        if not targets_path.exists():
            raise FileNotFoundError(
                f"Targets V13 Smart Crops ({split}) non trouvées.\n"
                f"Fichier manquant: {targets_path}\n\n"
                f"Lancez d'abord:\n"
                f"  python scripts/preprocessing/prepare_v13_smart_crops.py --family {family}"
            )

        print(f"\n🏷️ Famille: {family} ({split})")
        print(f"   Organes: {', '.join(get_organs(family))}")
        print(f"   Description: {FAMILY_DESCRIPTIONS[family]}")

        # Charger features
        print(f"\nChargement {features_path.name}...")
        features_data = np.load(features_path)
        self.features = features_data['features']  # (N_crops, 261, 1536)
        self.source_image_ids = features_data['source_image_ids']
        self.crop_positions = features_data['crop_positions']

        n_crops = len(self.features)
        n_source_images = len(np.unique(self.source_image_ids))
        print(f"  → {n_crops} crops depuis {n_source_images} images sources")
        print(f"  → Amplification: {n_crops / n_source_images:.1f}×")
        print(f"  → RAM: {self.features.nbytes / 1e9:.2f} GB")

        # Charger targets
        print(f"Chargement {targets_path.name}...")
        targets_data = np.load(targets_path)
        self.np_targets = targets_data['np_targets']  # (N_crops, 224, 224)
        self.hv_targets = targets_data['hv_targets']  # (N_crops, 2, 224, 224)
        self.nt_targets = targets_data['nt_targets']  # (N_crops, 224, 224)

        # Weight maps (Ronneberger) - optionnel, pour sur-pondérer frontières
        if 'weight_maps' in targets_data:
            self.weight_maps = targets_data['weight_maps']  # (N_crops, 224, 224)
            print(f"  ✅ Weight maps chargées: shape {self.weight_maps.shape}")
            print(f"     Range: [{self.weight_maps.min():.2f}, {self.weight_maps.max():.2f}]")
        else:
            self.weight_maps = None
            print(f"  ⚠️  Weight maps non trouvées - utilisation poids uniforme")

        # Images RGB pour mode hybride (injection H-channel)
        if use_hybrid:
            if 'images' in targets_data:
                self.images = targets_data['images']  # (N_crops, 224, 224, 3) uint8
                print(f"  ✅ Images RGB chargées: shape {self.images.shape}, dtype {self.images.dtype}")
                print(f"     Range: [{self.images.min()}, {self.images.max()}]")
            else:
                raise ValueError(
                    f"Mode hybride activé mais 'images' non trouvées dans {targets_path}.\n"
                    f"Régénérez les données avec:\n"
                    f"  python scripts/preprocessing/prepare_v13_smart_crops.py --family {family}"
                )
        else:
            self.images = None

        # Validation HV
        print(f"\nValidation HV targets:")
        print(f"  Dtype: {self.hv_targets.dtype}")
        print(f"  Range: [{self.hv_targets.min():.3f}, {self.hv_targets.max():.3f}]")

        if self.hv_targets.dtype != np.float32:
            print(f"  ⚠️  WARNING: HV dtype devrait être float32")
        if not (-1.0 <= self.hv_targets.min() <= self.hv_targets.max() <= 1.0):
            print(f"  ⚠️  WARNING: HV range devrait être [-1, 1]")

        total_targets_gb = (
            self.np_targets.nbytes + self.hv_targets.nbytes + self.nt_targets.nbytes
        ) / 1e9
        print(f"  → Targets: {total_targets_gb:.2f} GB")

        print(f"\n📊 Dataset {family} ({split}): {n_crops} crops (tout en RAM)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx].copy()
        np_target = self.np_targets[idx].copy()
        hv_target = self.hv_targets[idx].copy()
        nt_target = self.nt_targets[idx].copy()

        # Weight map (Ronneberger) - optionnel
        if self.weight_maps is not None:
            weight_map = self.weight_maps[idx].copy()
        else:
            weight_map = np.ones_like(np_target, dtype=np.float32)

        # Image RGB pour mode hybride
        if self.use_hybrid:
            image = self.images[idx].copy()  # (224, 224, 3) uint8
        else:
            image = None

        # Pas de resize nécessaire (déjà à 224×224)

        # Augmentation ALIGNÉE (2025-12-28): MÊME transformation pour features ET image
        if self.augmenter is not None and self.split == "train":
            features, np_target, hv_target, nt_target, weight_map, image = self.augmenter(
                features, np_target, hv_target, nt_target, weight_map, image
            )
            # L'image EST maintenant augmentée avec la même transformation que features
            # → Alignement spatial garanti entre H-channel et targets

        features = torch.from_numpy(features)
        np_target = torch.from_numpy(np_target.copy())
        hv_target = torch.from_numpy(hv_target.copy())
        nt_target = torch.from_numpy(nt_target.copy()).long()
        weight_map = torch.from_numpy(weight_map.copy())

        if self.use_hybrid:
            # Convertir image HWC uint8 → CHW float32 [0, 255]
            image = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
            return features, np_target, hv_target, nt_target, weight_map, image
        else:
            return features, np_target, hv_target, nt_target, weight_map


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calcule le Dice score pour NP en PER-SAMPLE puis moyenne.

    IMPORTANT: Utilise la même méthode que test_v13_smart_crops_aji.py
    pour garantir la cohérence des métriques training/test.

    Avant (Batch-wide): Samples avec beaucoup de noyaux dominent → Dice gonflé
    Après (Per-sample): Chaque sample a le même poids → Dice réaliste
    """
    pred_binary = (pred.argmax(dim=1) == 1).float()  # (B, H, W)
    target_float = target.float()  # (B, H, W)

    batch_size = pred_binary.shape[0]
    dice_scores = []

    for i in range(batch_size):
        pred_i = pred_binary[i]  # (H, W)
        target_i = target_float[i]  # (H, W)

        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum()

        if union == 0:
            dice_scores.append(1.0)
        else:
            dice_scores.append((2 * intersection / union).item())

    return sum(dice_scores) / len(dice_scores)


def compute_hv_mse(hv_pred: torch.Tensor, hv_target: torch.Tensor, np_target: torch.Tensor) -> float:
    """
    Calcule le MSE des cartes HV sur pixels de noyaux en PER-SAMPLE puis moyenne.

    Cohérent avec compute_dice() pour éviter que les samples denses dominent.
    """
    batch_size = hv_pred.shape[0]
    mse_scores = []

    for i in range(batch_size):
        mask_i = np_target[i].float()  # (H, W)

        if mask_i.sum() == 0:
            mse_scores.append(0.0)
            continue

        hv_pred_i = hv_pred[i]  # (2, H, W)
        hv_target_i = hv_target[i]  # (2, H, W)

        # Masquer uniquement les pixels de noyaux
        mask_expanded = mask_i.unsqueeze(0)  # (1, H, W)
        hv_pred_masked = hv_pred_i * mask_expanded
        hv_target_masked = hv_target_i * mask_expanded

        mse = ((hv_pred_masked - hv_target_masked) ** 2).sum() / mask_i.sum()
        mse_scores.append(mse.item())

    return sum(mse_scores) / len(mse_scores) if mse_scores else 0.0


def compute_nt_accuracy(nt_pred: torch.Tensor, nt_target: torch.Tensor, np_target: torch.Tensor) -> float:
    """
    Calcule l'accuracy de classification NT sur pixels de noyaux en PER-SAMPLE puis moyenne.

    Cohérent avec compute_dice() pour éviter que les samples denses dominent.
    """
    batch_size = nt_pred.shape[0]
    acc_scores = []

    for i in range(batch_size):
        mask_i = np_target[i] > 0  # (H, W)

        if mask_i.sum() == 0:
            acc_scores.append(1.0)  # Pas d'erreur possible si pas de noyaux
            continue

        pred_class_i = nt_pred[i].argmax(dim=0)  # (H, W)
        target_i = nt_target[i]  # (H, W)

        correct = (pred_class_i == target_i) & mask_i
        accuracy = correct.sum().float() / mask_i.sum().float()
        acc_scores.append(accuracy.item())

    return sum(acc_scores) / len(acc_scores) if acc_scores else 1.0


def train_one_epoch(
    model, dataloader, criterion, optimizer, device, epoch, n_classes, use_hybrid=False,
    debug_gradients=False
):
    """
    Train pour une epoch AVEC pondération spatiale Ronneberger.

    Les Weight Maps sur-pénalisent les erreurs aux frontières inter-cellulaires
    pour forcer le modèle à apprendre des séparations nettes.

    Args:
        use_hybrid: Si True, passe les images RGB au modèle pour injection H-channel.
        debug_gradients: Si True, affiche debug gradient H-channel (epoch 1).
    """
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_hv_mse = 0.0
    total_nt_acc = 0.0
    n_batches = 0

    # Gradient monitoring (V13-Hybrid-Final 2025-12-28)
    h_grad_magnitudes = [] if debug_gradients else None
    feature_grad_magnitudes = [] if debug_gradients else None

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Unpack batch selon mode hybride ou non
        if use_hybrid:
            features, np_target, hv_target, nt_target, weight_map, images = batch
            images = images.to(device)
        else:
            features, np_target, hv_target, nt_target, weight_map = batch
            images = None

        features = features.to(device)
        np_target = np_target.to(device)
        hv_target = hv_target.to(device)
        nt_target = nt_target.to(device)
        weight_map = weight_map.to(device)

        # Forward (avec images si mode hybride)
        np_out, hv_out, nt_out = model(features, images_rgb=images)

        # Loss avec pondération spatiale Ronneberger + Edge-Aware Loss
        # Edge-Aware Loss (Expert 2025-12-28): Sobel(H-channel) aligne HV magnitude
        loss, loss_dict = criterion(
            np_out, hv_out, nt_out,
            np_target, hv_target, nt_target,
            weight_map=weight_map,
            images_rgb=images  # Pour Edge-Aware Loss (extraction H-channel via Ruifrok)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient monitoring (V13-Hybrid-Final 2025-12-28)
        # Vérifie que le H-channel est "vivant" (reçoit du gradient)
        if debug_gradients and n_batches < 10:  # Premier 10 batches
            # Gradient sur la première couche de hv_head (DecoderHead.head[0])
            if hasattr(model, 'hv_head') and hasattr(model.hv_head, 'head'):
                first_conv = model.hv_head.head[0]
                if hasattr(first_conv, 'weight') and first_conv.weight.grad is not None:
                    hv_grad = first_conv.weight.grad.norm().item()
                else:
                    hv_grad = 0.0
            else:
                hv_grad = 0.0

            # Gradient sur bottleneck (features RGB) - bottleneck[0] est Conv2d
            if hasattr(model, 'bottleneck') and hasattr(model.bottleneck[0], 'weight'):
                if model.bottleneck[0].weight.grad is not None:
                    feat_grad = model.bottleneck[0].weight.grad.norm().item()
                else:
                    feat_grad = 0.0
            else:
                feat_grad = 0.0

            # Gradient sur H-channel (si mode hybride ou FPN Chimique)
            # V4: Monitorer FPN Chimique (h_pyramid) OU mode hybride classique (ruifrok + h_projection)
            h_norm_grad = 0.0
            h_proj_grad = 0.0
            if use_hybrid:
                # Mode FPN Chimique: vérifier h_pyramid.projections
                if hasattr(model, 'h_pyramid') and model.h_pyramid is not None:
                    # Prendre la projection au niveau 16×16 comme référence
                    proj_16 = model.h_pyramid.projections['16'][0]  # Premier Conv2d
                    if hasattr(proj_16, 'weight') and proj_16.weight.grad is not None:
                        h_proj_grad = proj_16.weight.grad.norm().item()
                    # Aussi vérifier niveau 224 (final)
                    proj_224 = model.h_pyramid.projections['224'][0]
                    if hasattr(proj_224, 'weight') and proj_224.weight.grad is not None:
                        h_norm_grad = proj_224.weight.grad.norm().item()  # Réutilise variable pour affichage
                else:
                    # Mode hybride classique (sans FPN)
                    # InstanceNorm dans RuifrokExtractor
                    if hasattr(model, 'ruifrok') and model.ruifrok is not None and hasattr(model.ruifrok, 'instance_norm'):
                        if model.ruifrok.instance_norm.weight is not None and model.ruifrok.instance_norm.weight.grad is not None:
                            h_norm_grad = model.ruifrok.instance_norm.weight.grad.norm().item()

                    # Projection 1→16 canaux (V3)
                    if hasattr(model, 'h_projection') and model.h_projection is not None:
                        # h_projection est un Sequential, prendre le premier Conv2d
                        if hasattr(model.h_projection[0], 'weight') and model.h_projection[0].weight.grad is not None:
                            h_proj_grad = model.h_projection[0].weight.grad.norm().item()

                h_total_grad = h_norm_grad + h_proj_grad
                h_grad_magnitudes.append(h_total_grad)

            feature_grad_magnitudes.append(feat_grad)

            if n_batches == 0:
                print(f"\n🔍 GRADIENT DEBUG (Epoch 1, Batch {n_batches+1}):")
                print(f"   HV Head grad norm:       {hv_grad:.6f}")
                print(f"   Bottleneck grad norm:    {feat_grad:.6f}")
                if use_hybrid:
                    # Affichage adapté au mode FPN Chimique
                    if hasattr(model, 'h_pyramid') and model.h_pyramid is not None:
                        print(f"   🧪 FPN Chimique H@16 grad:  {h_proj_grad:.6f}")
                        print(f"   🧪 FPN Chimique H@224 grad: {h_norm_grad:.6f}")
                    else:
                        print(f"   H-channel InstanceNorm:  {h_norm_grad:.6f}")
                        print(f"   H-channel Projection:    {h_proj_grad:.6f}")
                    h_total = h_norm_grad + h_proj_grad
                    if h_total < 1e-6:
                        print(f"   ⚠️  ALERTE: H-channel gradient ≈ 0 → Passager Clandestin!")
                    elif h_total < feat_grad * 0.05:
                        print(f"   ⚠️  ALERTE: H-channel gradient < 5% features → sous-exploité")
                    else:
                        print(f"   ✅ H-channel gradient OK (ratio: {h_total/feat_grad:.2%})")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Métriques
        dice = compute_dice(np_out, np_target)
        hv_mse = compute_hv_mse(hv_out, hv_target, np_target)
        nt_acc = compute_nt_accuracy(nt_out, nt_target, np_target)

        total_loss += loss.item()
        total_dice += dice
        total_hv_mse += hv_mse
        total_nt_acc += nt_acc
        n_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}',
            'hv_mse': f'{hv_mse:.4f}'
        })

    # Résumé gradient debug (epoch 1)
    if debug_gradients and feature_grad_magnitudes:
        print(f"\n📊 GRADIENT SUMMARY (Epoch 1, {len(feature_grad_magnitudes)} batches):")
        mean_feat = sum(feature_grad_magnitudes) / len(feature_grad_magnitudes)
        print(f"   Feature gradient mean:   {mean_feat:.6f}")
        if h_grad_magnitudes:
            mean_h = sum(h_grad_magnitudes) / len(h_grad_magnitudes)
            print(f"   H-channel gradient mean: {mean_h:.6f}")
            ratio = mean_h / mean_feat if mean_feat > 0 else 0
            print(f"   Ratio H/Features:        {ratio:.2%}")
            if ratio < 0.01:
                print(f"   ⚠️  DIAGNOSTIC: H-channel sous-exploité → InstanceNorm peut aider")
            elif ratio > 0.5:
                print(f"   ✅ DIAGNOSTIC: H-channel bien intégré")
            else:
                print(f"   ℹ️  DIAGNOSTIC: H-channel contribue modérément")

    return {
        'loss': total_loss / n_batches,
        'dice': total_dice / n_batches,
        'hv_mse': total_hv_mse / n_batches,
        'nt_acc': total_nt_acc / n_batches
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, n_classes, use_hybrid=False):
    """
    Validation SANS pondération spatiale.

    IMPORTANT (Tech Lead 2025-12-28):
    - Weight Maps = outil d'apprentissage (guide le gradient en TRAIN)
    - VAL doit mesurer l'erreur RÉELLE (conditions d'inférence)
    - Sinon: Loss VAL artificiellement gonflée → overfitting difficile à détecter
    - AJI/Dice ne sont JAMAIS pondérés (comparaison masques binaires purs)

    Args:
        use_hybrid: Si True, passe les images RGB au modèle pour injection H-channel.
    """
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_hv_mse = 0.0
    total_nt_acc = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validation"):
        # Unpack batch selon mode hybride ou non
        if use_hybrid:
            features, np_target, hv_target, nt_target, weight_map, images = batch
            images = images.to(device)
        else:
            features, np_target, hv_target, nt_target, weight_map = batch
            images = None

        features = features.to(device)
        np_target = np_target.to(device)
        hv_target = hv_target.to(device)
        nt_target = nt_target.to(device)
        # weight_map ignoré en validation - poids uniforme implicite (weight_map=None)

        # Forward (avec images si mode hybride)
        np_out, hv_out, nt_out = model(features, images_rgb=images)

        # Loss SANS pondération (conditions réelles d'inférence)
        # Note: Edge-Aware Loss toujours active pour évaluation cohérente
        loss, loss_dict = criterion(
            np_out, hv_out, nt_out,
            np_target, hv_target, nt_target,
            weight_map=None,  # ← Poids uniforme pour VAL
            images_rgb=images  # Pour Edge-Aware Loss
        )

        # Métriques
        dice = compute_dice(np_out, np_target)
        hv_mse = compute_hv_mse(hv_out, hv_target, np_target)
        nt_acc = compute_nt_accuracy(nt_out, nt_target, np_target)

        total_loss += loss.item()
        total_dice += dice
        total_hv_mse += hv_mse
        total_nt_acc += nt_acc
        n_batches += 1

        # Diagnostic NT sur premier batch de validation
        if n_batches == 1:
            mask = np_target > 0
            print(f"\n🔍 DIAGNOSTIC NT (Validation - Batch 1):")
            print(f"  nt_out shape: {nt_out.shape}")
            print(f"  nt_target shape: {nt_target.shape}")
            print(f"  nt_target unique: {torch.unique(nt_target)}")
            print(f"  nt_target (noyaux) unique: {torch.unique(nt_target[mask]) if mask.sum() > 0 else 'no nuclei'}")
            print(f"  nt_pred argmax unique: {torch.unique(nt_out.argmax(dim=1))}")
            print(f"  nt_pred argmax (noyaux) unique: {torch.unique(nt_out.argmax(dim=1)[mask]) if mask.sum() > 0 else 'no nuclei'}")
            print(f"  mask sum: {mask.sum().item()} / {mask.numel()} pixels")
            print(f"  NT Acc: {nt_acc:.4f}\n")

    return {
        'loss': total_loss / n_batches,
        'dice': total_dice / n_batches,
        'hv_mse': total_hv_mse / n_batches,
        'nt_acc': total_nt_acc / n_batches
    }


def main():
    parser = argparse.ArgumentParser(
        description="Entraînement HoVer-Net V13 Smart Crops"
    )
    parser.add_argument(
        "--family",
        required=True,
        choices=FAMILIES,
        help="Famille d'organes"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_np", type=float, default=1.0)
    parser.add_argument("--lambda_hv", type=float, default=2.0,
                       help="Poids branche HV initial (Phase 1: convergence Dice/NT)")
    parser.add_argument("--lambda_hv_boost", type=float, default=8.0,
                       help="Poids branche HV boost (Phase 2: fine-tuning séparation)")
    parser.add_argument("--lambda_hv_boost_ratio", type=float, default=0.85,
                       help="Ratio d'epochs avant boost λ_hv (0.85 = boost aux derniers 15%%)")
    parser.add_argument("--lambda_nt", type=float, default=1.0)
    parser.add_argument("--lambda_magnitude", type=float, default=1.0,
                       help="Poids magnitude loss (force gradients HV)")
    parser.add_argument("--lambda_edge", type=float, default=0.5,
                       help="Poids Edge-Aware Loss (Sobel sur H-channel) - Expert 2025-12-28")
    parser.add_argument("--dropout", type=float, default=0.4,
                       help="Dropout rate pour régularisation")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--adaptive_loss", action="store_true",
                       help="Activer Uncertainty Weighting (poids appris)")
    parser.add_argument("--use_hybrid", action="store_true",
                       help="Activer mode hybride RGB+H-channel (injection Hématoxyline)")
    parser.add_argument("--use_se_block", action="store_true",
                       help="Activer SE-Block pour recalibration attention sur H-channel (Hu et al. 2018)")
    parser.add_argument("--use_fpn_chimique", action="store_true",
                       help="Activer FPN Chimique (Multi-scale H-Injection) à 5 niveaux: 16, 32, 64, 112, 224")
    parser.add_argument("--use_h_instance_norm", action="store_true",
                       help="[DEPRECATED] InstanceNorm lisse les gradients HV - utiliser --use_h_alpha")
    parser.add_argument("--use_h_alpha", action="store_true",
                       help="Activer facteur α learnable pour amplifier H-channel (init=10.0 par niveau)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Chemin vers checkpoint pour reprendre l'entraînement (même famille)")

    # === TRANSFER LEARNING INTER-FAMILLE (Expert 2025-12-29) ===
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                       help="Checkpoint d'une AUTRE famille pour Transfer Learning (ex: Respiratory → Epidermal)")
    parser.add_argument("--finetune_lr", type=float, default=1e-5,
                       help="Learning rate ultra-bas pour fine-tuning inter-famille (évite catastrophic forgetting)")

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device)
    n_classes = 5  # PanNuke: 5 classes

    print("=" * 80)
    print("ENTRAÎNEMENT HOVERNET V13 SMART CROPS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Famille: {args.family}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Lambda (NP/HV/NT/Mag/Edge): {args.lambda_np}/{args.lambda_hv}/{args.lambda_nt}/{args.lambda_magnitude}/{args.lambda_edge}")

    # Calculer l'epoch de boost dynamiquement (adaptatif au nombre total d'epochs)
    # Sécurité: clamp ratio entre 0.0 et 1.0 (règle 20/12/2025 - priorité flux de code)
    ratio = max(0.0, min(1.0, args.lambda_hv_boost_ratio))
    lambda_hv_boost_epoch = int(args.epochs * ratio) + 1

    # Log de contrôle stratégie HV (visibilité production)
    print(f"  🚀 Stratégie HV : Phase 1 (λ={args.lambda_hv}) jusqu'à epoch {lambda_hv_boost_epoch-1}")
    print(f"  🔥 Stratégie HV : Phase 2 (λ={args.lambda_hv_boost}) à partir de epoch {lambda_hv_boost_epoch}")
    print(f"  Lambda HV boost ratio: {ratio} (adaptatif: {lambda_hv_boost_epoch}/{args.epochs})")
    print(f"  Dropout: {args.dropout}")
    print(f"  Augmentation: {args.augment}")
    print(f"  Adaptive loss: {args.adaptive_loss}")
    print(f"  Hybrid mode: {args.use_hybrid} (injection H-channel)")
    print(f"  SE-Block: {args.use_se_block} (attention recalibration)")
    print(f"  FPN Chimique: {args.use_fpn_chimique} (multi-scale H @ 16,32,64,112,224)")
    print(f"  Device: {args.device}")

    # Datasets
    print("\n" + "=" * 80)
    print("CHARGEMENT DATASETS")
    print("=" * 80)

    train_dataset = V13SmartCropsDataset(
        family=args.family,
        split="train",
        augment=args.augment,
        use_hybrid=args.use_hybrid
    )

    val_dataset = V13SmartCropsDataset(
        family=args.family,
        split="val",
        augment=False,  # Pas d'augmentation en validation
        use_hybrid=args.use_hybrid
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Modèle
    print("\n" + "=" * 80)
    print("INITIALISATION MODÈLE")
    print("=" * 80)

    model = HoVerNetDecoder(
        embed_dim=1536,
        n_classes=n_classes,
        dropout=args.dropout,
        use_hybrid=args.use_hybrid,
        use_se_block=args.use_se_block,
        use_fpn_chimique=args.use_fpn_chimique,
        use_h_instance_norm=args.use_h_instance_norm
    ).to(device)

    if args.use_hybrid:
        print(f"  → Mode HYBRID activé: injection H-channel via RuifrokExtractor")
        print(f"  → up1 input channels: 257 (256 bottleneck + 1 H-channel)")

    if args.use_se_block:
        print(f"  → SE-Block activé: recalibration des canaux post-fusion")
        print(f"  → Attention forcée sur H-channel aux pixels de frontière")

    if args.use_fpn_chimique:
        print(f"  → FPN CHIMIQUE activé: injection H-channel à 5 niveaux")
        print(f"  → Niveaux: 16×16 → 32×32 → 64×64 → 112×112 → 224×224")
        print(f"  → Architecture: Bottleneck(256) + H@16(16) = 272 → up1 → 128 + H@32(16) = 144 → ...")
        print(f"  → Objectif: Briser la 'Cécité Profonde' - H visible dès le niveau 0")

    if args.use_h_instance_norm:
        print(f"  → H-InstanceNorm activé: normalisation H-channel à mean=0, std=1")
        print(f"  → Objectif: Rééquilibrer la force statistique H vs Features")

    # Resume from checkpoint if provided (SAME family)
    start_epoch = 0
    effective_lr = args.lr  # Default LR
    is_transfer_learning = False

    if args.resume:
        print(f"\n  📥 Chargement checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"  ✅ Checkpoint chargé (epoch {start_epoch})")
        print(f"  → Fine-tuning pour {args.epochs} epochs supplémentaires")

    # Transfer Learning from DIFFERENT family checkpoint (Expert 2025-12-29)
    elif args.pretrained_checkpoint:
        is_transfer_learning = True
        effective_lr = args.finetune_lr  # Ultra-low LR for transfer learning

        print(f"\n  🔄 TRANSFER LEARNING INTER-FAMILLE")
        print(f"  📥 Checkpoint source: {args.pretrained_checkpoint}")

        checkpoint = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)

        # Load weights only (NOT optimizer state - we want fresh optimizer with low LR)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Extract source info from checkpoint name
        source_checkpoint = Path(args.pretrained_checkpoint).name
        source_epoch = checkpoint.get('epoch', 'unknown')
        source_dice = checkpoint.get('val_metrics', {}).get('dice', 'N/A')

        print(f"  ✅ Poids chargés depuis: {source_checkpoint}")
        print(f"  → Source epoch: {source_epoch}")
        print(f"  → Source Dice: {source_dice}")
        print(f"  → Reset epoch: 0 (nouveau départ pour famille {args.family})")
        print(f"  → LR ultra-bas: {effective_lr} (vs {args.lr} normal)")
        print(f"  → Objectif: Adapter les patterns Membrane/HV à la nouvelle famille")
        # start_epoch reste à 0 (nouveau départ)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  → Paramètres: {n_params:,}")

    # Loss
    criterion = HoVerNetLoss(
        lambda_np=args.lambda_np,
        lambda_hv=args.lambda_hv,
        lambda_nt=args.lambda_nt,
        lambda_magnitude=args.lambda_magnitude,
        lambda_edge=args.lambda_edge,  # Edge-Aware Loss (Expert 2025-12-28)
        adaptive=args.adaptive_loss
    )

    # Si adaptive loss, mettre criterion sur le device (contient des paramètres apprenables)
    if args.adaptive_loss:
        criterion.to(device)

    # Optimizer (inclut les paramètres de loss si adaptive)
    # Note: effective_lr = args.finetune_lr si Transfer Learning, sinon args.lr
    if args.adaptive_loss:
        optimizer = AdamW(
            list(model.parameters()) + list(criterion.parameters()),
            lr=effective_lr, weight_decay=1e-4
        )
    else:
        optimizer = AdamW(model.parameters(), lr=effective_lr, weight_decay=1e-4)

    if is_transfer_learning:
        print(f"\n  ⚙️ Optimizer configuré pour Transfer Learning:")
        print(f"  → LR: {effective_lr} (ultra-bas pour éviter catastrophic forgetting)")
        print(f"  → Weight decay: 1e-4")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training
    print("\n" + "=" * 80)
    print("ENTRAÎNEMENT")
    print("=" * 80)

    best_dice = 0.0
    best_combined_score = -float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_hv_mse': [],
        'val_nt_acc': []
    }

    # Checkpoints directory
    checkpoint_dir = PROJECT_ROOT / "models/checkpoints_v13_smart_crops"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        # Lambda HV Scheduler Adaptatif (Expert 2025-12-29)
        # Phase 1 (epochs 1 → ratio×total): lambda_hv stable pour convergence Dice/NT
        # Phase 2 (epochs ratio×total → fin): boost lambda_hv pour affiner séparation membranes
        # Exemple: 30 epochs × 0.85 = epoch 26+, 60 epochs × 0.85 = epoch 52+
        if epoch >= lambda_hv_boost_epoch:
            current_lambda_hv = args.lambda_hv_boost
        else:
            current_lambda_hv = args.lambda_hv

        # Mettre à jour lambda_hv dans criterion
        criterion.lambda_hv = current_lambda_hv

        print(f"\nEpoch {epoch}/{args.epochs} [lambda_hv={current_lambda_hv}]")

        # Train avec gradient monitoring epoch 1
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, n_classes,
            use_hybrid=args.use_hybrid,
            debug_gradients=(epoch == 1)  # Debug gradients epoch 1 uniquement
        )

        # Validation
        val_metrics = validate(
            model, val_loader, criterion, device, n_classes,
            use_hybrid=args.use_hybrid
        )

        # Scheduler step
        scheduler.step()

        # Logs
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, "
              f"HV MSE: {val_metrics['hv_mse']:.4f}, NT Acc: {val_metrics['nt_acc']:.4f}")

        # History
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_hv_mse'].append(val_metrics['hv_mse'])
        history['val_nt_acc'].append(val_metrics['nt_acc'])

        # Save best
        combined_score = val_metrics['dice'] - 0.5 * val_metrics['hv_mse']

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_dice = val_metrics['dice']

            # Suffix pour différencier modes
            suffix = ""
            if args.use_hybrid:
                suffix += "_hybrid"
            if args.use_fpn_chimique:
                suffix += "_fpn"
            checkpoint_path = checkpoint_dir / f"hovernet_{args.family}_v13_smart_crops{suffix}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'best_combined_score': best_combined_score,
                'val_metrics': val_metrics,
                'args': vars(args),
                'use_hybrid': args.use_hybrid,  # Important pour chargement inference
                'use_fpn_chimique': args.use_fpn_chimique  # FPN Chimique multi-scale
            }, checkpoint_path)

            print(f"✅ Best model saved (Score: {combined_score:.4f})")

    # Save history
    suffix = ""
    if args.use_hybrid:
        suffix += "_hybrid"
    if args.use_fpn_chimique:
        suffix += "_fpn"
    history_path = checkpoint_dir / f"hovernet_{args.family}_v13_smart_crops{suffix}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 80)
    print(f"\nBest Dice: {best_dice:.4f}")
    print(f"Best Combined Score: {best_combined_score:.4f}")
    checkpoint_final = checkpoint_dir / f"hovernet_{args.family}_v13_smart_crops{suffix}_best.pth"
    print(f"\nCheckpoint: {checkpoint_final}")
    print(f"History: {history_path}")
    print("\nProchaine étape:")
    flags = ""
    if args.use_hybrid:
        flags += " --use_hybrid"
    if args.use_fpn_chimique:
        flags += " --use_fpn_chimique"
    print(f"  python scripts/evaluation/test_v13_smart_crops_aji.py \\")
    print(f"      --family {args.family} --n_samples 50{flags}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
