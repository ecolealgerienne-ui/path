#!/usr/bin/env python3
"""
Préparation données V13 Smart Crops - Algorithme par couche avec split train/val.

⚠️ APPROCHE SPLIT-FIRST-THEN-ROTATE (CTO-validated):
1. Collecter toutes les images sources de la famille
2. Split train/val par source_image_ids (80/20) - ZÉRO data leakage
3. Pour chaque split, appliquer algorithme par couche:
   - Couche 1: TOUS les crops CENTRE
   - Couche 2: Si < max_samples → ajouter TOUS les TOP_LEFT + rotation
   - Couche 3: Si < max_samples → ajouter TOUS les TOP_RIGHT + rotation
   - etc. jusqu'à max_samples ou couches épuisées
4. Sauvegarder 2 fichiers: {family}_train_v13_smart_crops.npz et {family}_val_v13_smart_crops.npz

Usage:
    python scripts/preprocessing/prepare_v13_smart_crops.py \
        --family epidermal \
        --pannuke_dir /home/amar/data/PanNuke \
        --output_dir data/family_data_v13_smart_crops \
        --max_samples 5000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE
from src.models.organ_families import ORGAN_TO_FAMILY

# Positions de crop fixes (5 crops par image 256×256)
CROP_POSITIONS = {
    'center':       (16, 16, 240, 240),
    'top_left':     (0,  0,  224, 224),
    'top_right':    (32, 0,  256, 224),
    'bottom_left':  (0,  32, 224, 256),
    'bottom_right': (32, 32, 256, 256),
}

# Stratégie V13 Smart Crops: chaque crop a UNE rotation spécifique
# = 5 échantillons par image (pas 25!)
CROP_ROTATION_MAPPING = {
    'center':       '0',       # Référence sans rotation
    'top_left':     '90',      # 90° clockwise
    'top_right':    '180',     # 180°
    'bottom_left':  '270',     # 270° clockwise (= 90° CCW)
    'bottom_right': 'flip_h',  # Flip horizontal
}

# Ordre des couches pour algorithme de génération par couche
# Couche 1 = center (prioritaire), puis coins en ordre
LAYER_ORDER = ['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right']

CROP_SIZE = 224


def normalize_mask_format(mask: np.ndarray) -> np.ndarray:
    """Normalise format masque PanNuke (H, W, 6)."""
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=-1)
    return mask


def extract_pannuke_instances(mask: np.ndarray) -> np.ndarray:
    """
    Extrait instance map depuis masque PanNuke.

    Args:
        mask: Masque PanNuke (H, W, 6)

    Returns:
        inst_map: Instance map (H, W) int32
    """
    mask = normalize_mask_format(mask)

    h, w = mask.shape[:2]
    inst_map = np.zeros((h, w), dtype=np.int32)
    instance_counter = 1

    # Channel 0: Multi-type instances
    channel_0 = mask[:, :, 0]
    if channel_0.max() > 0:
        inst_ids_0 = np.unique(channel_0)
        inst_ids_0 = inst_ids_0[inst_ids_0 > 0]

        for inst_id in inst_ids_0:
            inst_mask = channel_0 == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Canaux 1-4: Class-specific instances
    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        if channel_mask.max() > 0:
            inst_ids = np.unique(channel_mask)
            inst_ids = inst_ids[inst_ids > 0]

            for inst_id in inst_ids:
                inst_mask = channel_mask == inst_id
                inst_mask_new = inst_mask & (inst_map == 0)

                if inst_mask_new.sum() > 0:
                    inst_map[inst_mask_new] = instance_counter
                    instance_counter += 1

    return inst_map


def compute_hv_maps(inst_map: np.ndarray) -> np.ndarray:
    """
    Calcule cartes HV (Horizontal/Vertical) centripètes via Distance Transform.

    AMÉLIORATION 2025-12-28: Utilise Distance Transform au lieu du centroïde.
    FIX CRITIQUE 2025-12-28: Normalisation ISOTROPE pour préserver l'aspect ratio.

    Problème avec centroïde (mean):
    - Pour les noyaux concaves (forme de C, rein), le centroïde peut
      tomber EN DEHORS du noyau, dans le vide/background.
    - Les vecteurs HV pointent vers un centre inexistant.
    - Le Watershed ne trouve pas de "pic d'énergie" → instances perdues.

    Solution Distance Transform:
    - Le centre est le pixel le plus ÉLOIGNÉ des bords (le plus "profond").
    - Garantit que le centre est TOUJOURS à l'intérieur du noyau.
    - Robuste pour toutes les formes (rondes, allongées, concaves).

    FIX Normalisation Isotrope:
    - AVANT: H normalisé par max_dist_x, V par max_dist_y → gradients "déformés"
    - APRÈS: H et V normalisés par max(max_dist_x, max_dist_y) → géométrie préservée
    - Résultat: Pics Watershed nets au lieu de traînées floues → AJI amélioré

    Args:
        inst_map: Instance map (H, W) int32

    Returns:
        hv_map: (2, H, W) float32 [-1, 1]
    """
    from scipy.ndimage import distance_transform_edt

    h, w = inst_map.shape
    hv_map = np.zeros((2, h, w), dtype=np.float32)

    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    for inst_id in inst_ids:
        inst_mask = inst_map == inst_id
        y_coords, x_coords = np.where(inst_mask)

        if len(y_coords) == 0:
            continue

        # Distance Transform: trouver le pixel le plus éloigné des bords
        # C'est le "vrai centre biologique" du noyau
        dist_map = distance_transform_edt(inst_mask)
        idx_max = np.argmax(dist_map)
        cy, cx = np.unravel_index(idx_max, inst_mask.shape)

        # Distances brutes (centripètes)
        h_dist = cx - x_coords  # Horizontal
        v_dist = cy - y_coords  # Vertical

        # NORMALISATION ISOTROPE (Fix critique)
        # Utilise la MÊME valeur max pour H et V → préserve l'aspect ratio
        max_dist_x = np.abs(h_dist).max() if len(h_dist) > 0 else 1e-6
        max_dist_y = np.abs(v_dist).max() if len(v_dist) > 0 else 1e-6
        global_max_dist = max(max_dist_x, max_dist_y, 1e-6)

        # Vecteurs centripètes normalisés [-1, 1]
        # Convention HoVer-Net: vecteurs pointent VERS le centre
        h_map = h_dist / global_max_dist
        v_map = v_dist / global_max_dist

        # Convention HoVer-Net: H à index 0, V à index 1
        hv_map[0, y_coords, x_coords] = h_map  # Horizontal
        hv_map[1, y_coords, x_coords] = v_map  # Vertical

    return hv_map


def compute_spatial_weight_map(inst_map: np.ndarray, w0: float = 10.0, sigma: float = 5.0) -> np.ndarray:
    """
    Génère une carte de poids pour sur-pondérer les bordures entre noyaux.

    Méthode Ronneberger (U-Net 2015): Les pixels aux frontières inter-cellulaires
    reçoivent un poids plus élevé pour forcer le modèle à apprendre des séparations nettes.

    Formule: weight = 1 + w0 * exp(-(d1 + d2)² / (2 * sigma²))

    où d1 = distance au noyau le plus proche
       d2 = distance au deuxième noyau le plus proche

    Args:
        inst_map: Instance map (H, W) int32
        w0: Poids additionnel pour les zones de contact (défaut: 10x)
        sigma: Étendue de l'influence du poids en pixels (défaut: 5)

    Returns:
        weight_map: (H, W) float32, valeurs >= 1.0
    """
    from scipy.ndimage import distance_transform_edt

    # Si moins de 2 noyaux, pas besoin de pondération spéciale
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    if len(inst_ids) < 2:
        return np.ones(inst_map.shape, dtype=np.float32)

    h, w = inst_map.shape

    # Calculer la distance à chaque instance séparément
    all_distances = []
    for inst_id in inst_ids:
        mask = (inst_map == inst_id)
        # Distance depuis l'EXTÉRIEUR du noyau vers le noyau
        dist = distance_transform_edt(~mask)
        all_distances.append(dist)

    all_distances = np.stack(all_distances, axis=0)  # (N_instances, H, W)
    all_distances = np.sort(all_distances, axis=0)   # Trier par distance

    # d1: distance au noyau le plus proche
    # d2: distance au deuxième noyau le plus proche
    d1 = all_distances[0]
    d2 = all_distances[1]

    # Formule de Ronneberger : poids élevé là où (d1 + d2) est petit
    # = zones entre deux noyaux proches
    weight_map = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))

    # Ajouter 1 pour garder un poids normal partout ailleurs
    weight_map = (1.0 + weight_map).astype(np.float32)

    return weight_map


def compute_np_target(mask: np.ndarray) -> np.ndarray:
    """Génère target NP (Nuclear Presence) binaire."""
    nuclei_mask = mask[:, :, :5].sum(axis=-1) > 0
    return nuclei_mask.astype(np.float32)


def compute_nt_target(mask: np.ndarray) -> np.ndarray:
    """
    Génère target NT (Nuclear Type) multiclass.

    Args:
        mask: PanNuke mask (H, W, 6) avec canaux:
            - Canal 0: Background (ignoré)
            - Canal 1: Neoplastic → classe 0
            - Canal 2: Inflammatory → classe 1
            - Canal 3: Connective → classe 2
            - Canal 4: Dead → classe 3
            - Canal 5: Epithelial → classe 4

    Returns:
        nt_target: (H, W) int64 avec valeurs 0-4 pour les 5 types cellulaires
    """
    nt_target = np.zeros(mask.shape[:2], dtype=np.int64)

    # Assigner chaque type cellulaire (priorité: dernier canal écrase précédents si overlap)
    for c in range(5):
        type_mask = mask[:, :, c + 1] > 0
        nt_target[type_mask] = c

    return nt_target


def extract_raw_crop(
    image: np.ndarray,
    np_target: np.ndarray,
    nt_target: np.ndarray,
    x1: int, y1: int, x2: int, y2: int
) -> Dict[str, np.ndarray]:
    """
    Extrait un crop 224×224 SANS labeling ni HV (sera fait APRÈS rotation).

    ✅ FIX Expert 2025-12-28: label() et compute_hv_maps() APRÈS rotation

    PROBLÈME IDENTIFIÉ:
    - label() numérote les noyaux de gauche à droite
    - Si rotation APRÈS labeling → ordre des IDs incohérent avec image rotée
    - Pour Transformer (H-optimus-0): ordre spatial des tokens crucial
    - Cette dissonance crée du "bruit" qui bloque la précision à ~0.55 AJI

    SOLUTION:
    1. extract_raw_crop() → juste slice (image, np, nt)
    2. apply_rotation() → rotation spatiale
    3. finalize_crop_after_rotation() → label() + compute_hv_maps() sur image rotée

    Résultat: IDs d'instances suivent l'ordre naturel dans l'image FINALE.

    Args:
        image: Image RGB 256×256
        np_target: Masque binaire 256×256
        nt_target: Types cellulaires 256×256
        x1, y1, x2, y2: Coordonnées crop

    Returns:
        Dict avec crop RAW (sans inst_map ni hv_target)
    """
    crop_image = image[y1:y2, x1:x2]
    crop_np = np_target[y1:y2, x1:x2]
    crop_nt = nt_target[y1:y2, x1:x2]

    # Validation basique
    assert crop_image.shape == (CROP_SIZE, CROP_SIZE, 3), f"Image shape: {crop_image.shape}"
    assert crop_np.shape == (CROP_SIZE, CROP_SIZE), f"NP shape: {crop_np.shape}"
    assert crop_nt.shape == (CROP_SIZE, CROP_SIZE), f"NT shape: {crop_nt.shape}"

    return {
        'image': crop_image,
        'np_target': crop_np,
        'nt_target': crop_nt,
    }


def apply_simple_rotation(
    image: np.ndarray,
    np_target: np.ndarray,
    nt_target: np.ndarray,
    rotation: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applique rotation SIMPLE (sans HV - car HV sera calculé après).

    Args:
        image: (224, 224, 3)
        np_target: (224, 224)
        nt_target: (224, 224)
        rotation: '0', '90', '180', '270', 'flip_h'

    Returns:
        (image_rot, np_rot, nt_rot)
    """
    if rotation == '0':
        return image, np_target, nt_target

    elif rotation == '90':
        image_rot = np.rot90(image, k=-1, axes=(0, 1))
        np_rot = np.rot90(np_target, k=-1, axes=(0, 1))
        nt_rot = np.rot90(nt_target, k=-1, axes=(0, 1))
        return image_rot, np_rot, nt_rot

    elif rotation == '180':
        image_rot = np.rot90(image, k=2, axes=(0, 1))
        np_rot = np.rot90(np_target, k=2, axes=(0, 1))
        nt_rot = np.rot90(nt_target, k=2, axes=(0, 1))
        return image_rot, np_rot, nt_rot

    elif rotation == '270':
        image_rot = np.rot90(image, k=1, axes=(0, 1))
        np_rot = np.rot90(np_target, k=1, axes=(0, 1))
        nt_rot = np.rot90(nt_target, k=1, axes=(0, 1))
        return image_rot, np_rot, nt_rot

    elif rotation == 'flip_h':
        image_rot = np.fliplr(image)
        np_rot = np.fliplr(np_target)
        nt_rot = np.fliplr(nt_target)
        return image_rot, np_rot, nt_rot

    else:
        raise ValueError(f"Unknown rotation: {rotation}")


def finalize_crop_after_rotation(
    image: np.ndarray,
    np_target: np.ndarray,
    nt_target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Finalise un crop APRÈS rotation: label() + compute_hv_maps() + weight_map.

    ✅ FIX Expert 2025-12-28: Cette fonction est appelée APRÈS rotation
    pour garantir que les IDs d'instances suivent l'ordre spatial naturel
    dans l'image FINALE (rotée).

    ✅ AJOUT Tech Lead 2025-12-28: Génère aussi la weight_map (Ronneberger)
    pour sur-pondérer les frontières inter-cellulaires.

    Args:
        image: Image rotée (224, 224, 3)
        np_target: Masque binaire roté (224, 224)
        nt_target: Types cellulaires rotés (224, 224)

    Returns:
        (hv_target, inst_map, weight_map, n_instances, is_valid)
    """
    from scipy.ndimage import label

    # Labeling sur image ROTÉE → IDs suivent ordre naturel gauche→droite
    binary_mask = (np_target > 0.5).astype(np.uint8)
    inst_map, n_instances = label(binary_mask)

    # Vérifier si crop valide (au moins 1 noyau)
    is_valid = n_instances > 0

    if not is_valid:
        # Retourner HV et weight_map vides si pas de noyaux
        hv_target = np.zeros((2, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        weight_map = np.ones((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        return hv_target, inst_map, weight_map, n_instances, False

    # Calculer HV sur image ROTÉE → vecteurs pointent vers bons centres
    hv_target = compute_hv_maps(inst_map)

    # Validation HV range
    assert hv_target.min() >= -1.0 and hv_target.max() <= 1.0, \
        f"HV range invalid: [{hv_target.min():.3f}, {hv_target.max():.3f}]"

    # Calculer weight_map (Ronneberger) pour sur-pondérer les frontières
    weight_map = compute_spatial_weight_map(inst_map, w0=10.0, sigma=5.0)

    return hv_target, inst_map, weight_map, n_instances, is_valid


def is_valid_crop(np_target: np.ndarray, nt_target: np.ndarray) -> Tuple[bool, int]:
    """Vérifie si un crop contient au moins 1 instance."""
    binary_mask = (np_target > 0.5).astype(np.uint8)
    inst_map, num_instances = ndimage.label(binary_mask)
    unique_labels = np.unique(inst_map)
    is_valid = len(unique_labels) > 1
    return is_valid, num_instances


def apply_rotation(
    image: np.ndarray,
    np_target: np.ndarray,
    hv_target: np.ndarray,
    nt_target: np.ndarray,
    inst_map: np.ndarray,
    rotation: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applique rotation déterministe avec HV component swapping.

    Args:
        image: (224, 224, 3)
        np_target: (224, 224)
        hv_target: (2, 224, 224) - [H, V] (Convention HoVer-Net)
        nt_target: (224, 224)
        inst_map: (224, 224) int32
        rotation: '0', '90', '180', '270', 'flip_h'

    Returns:
        (image_rot, np_rot, hv_rot, nt_rot, inst_map_rot)
    """
    if rotation == '0':
        return image, np_target, hv_target, nt_target, inst_map

    elif rotation == '90':
        # Rotation 90° clockwise
        image_rot = np.rot90(image, k=-1, axes=(0, 1))
        np_rot = np.rot90(np_target, k=-1, axes=(0, 1))
        nt_rot = np.rot90(nt_target, k=-1, axes=(0, 1))
        inst_map_rot = np.rot90(inst_map, k=-1, axes=(0, 1))

        # CORRECTION MATHÉMATIQUE (90° CW, référentiel image Y vers le bas)
        # Vecteur (1,0) droite → (0,1) bas après 90° CW
        # Formule: H' = -V, V' = H
        h_rot = -np.rot90(hv_target[1], k=-1, axes=(0, 1))  # H' = -V
        v_rot = np.rot90(hv_target[0], k=-1, axes=(0, 1))   # V' = H
        hv_rot = np.stack([h_rot, v_rot], axis=0)

        return image_rot, np_rot, hv_rot, nt_rot, inst_map_rot

    elif rotation == '180':
        # Rotation 180°
        image_rot = np.rot90(image, k=2, axes=(0, 1))
        np_rot = np.rot90(np_target, k=2, axes=(0, 1))
        nt_rot = np.rot90(nt_target, k=2, axes=(0, 1))
        inst_map_rot = np.rot90(inst_map, k=2, axes=(0, 1))

        # HV negation: H' = -H, V' = -V
        h_rot = -np.rot90(hv_target[0], k=2, axes=(0, 1))  # H' = -H
        v_rot = -np.rot90(hv_target[1], k=2, axes=(0, 1))  # V' = -V
        hv_rot = np.stack([h_rot, v_rot], axis=0)

        return image_rot, np_rot, hv_rot, nt_rot, inst_map_rot

    elif rotation == '270':
        # Rotation 270° clockwise (= 90° counter-clockwise)
        image_rot = np.rot90(image, k=1, axes=(0, 1))
        np_rot = np.rot90(np_target, k=1, axes=(0, 1))
        nt_rot = np.rot90(nt_target, k=1, axes=(0, 1))
        inst_map_rot = np.rot90(inst_map, k=1, axes=(0, 1))

        # CORRECTION MATHÉMATIQUE (270° CW, référentiel image Y vers le bas)
        # Vecteur (1,0) droite → (0,-1) haut après 270° CW
        # Formule: H' = V, V' = -H
        h_rot = np.rot90(hv_target[1], k=1, axes=(0, 1))   # H' = V
        v_rot = -np.rot90(hv_target[0], k=1, axes=(0, 1))  # V' = -H
        hv_rot = np.stack([h_rot, v_rot], axis=0)

        return image_rot, np_rot, hv_rot, nt_rot, inst_map_rot

    elif rotation == 'flip_h':
        # Flip horizontal
        image_rot = np.fliplr(image)
        np_rot = np.fliplr(np_target)
        nt_rot = np.fliplr(nt_target)
        inst_map_rot = np.fliplr(inst_map)

        # HV flip: H' = -H, V' = V
        h_rot = -np.fliplr(hv_target[0])  # H' = -H (négué car flip horizontal)
        v_rot = np.fliplr(hv_target[1])   # V' = V (inchangé)
        hv_rot = np.stack([h_rot, v_rot], axis=0)

        return image_rot, np_rot, hv_rot, nt_rot, inst_map_rot

    else:
        raise ValueError(f"Unknown rotation: {rotation}")


def generate_smart_crops_from_pannuke(
    pannuke_dir: Path,
    output_dir: Path,
    family: str,
    folds: list = None,
    max_samples: int = 5000,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Dict[str, int]:
    """
    Génère crops avec split train/val et algorithme par couche.

    APPROCHE SPLIT-FIRST-THEN-ROTATE:
    1. Collecter toutes les images sources
    2. Split train/val par source_image_ids (80/20)
    3. Pour chaque split, appliquer algorithme par couche:
       - Couche 1: TOUS les crops CENTRE
       - Couche 2: Si < max_samples → TOP_LEFT + rotation
       - etc. jusqu'à max_samples ou couches épuisées
    4. Sauvegarder 2 fichiers: {family}_train_v13_smart_crops.npz et {family}_val_v13_smart_crops.npz

    Args:
        pannuke_dir: Répertoire PanNuke
        output_dir: Répertoire de sortie
        family: Famille tissulaire
        folds: Liste des folds (défaut: [0, 1, 2])
        max_samples: Nombre maximum de samples PAR SPLIT (défaut: 5000)
        train_ratio: Ratio train/val (défaut: 0.8)
        seed: Seed pour reproductibilité

    Returns:
        Statistiques de génération
    """
    if folds is None:
        folds = [0, 1, 2]

    print(f"\n{'='*70}")
    print(f"GÉNÉRATION V13 SMART CROPS - Famille: {family.upper()}")
    print(f"{'='*70}")
    print(f"Max samples: {max_samples}")
    print(f"Algorithme: Par couche (center → top_left → top_right → ...)\n")

    # Organes de cette famille
    organs = [org for org, fam in ORGAN_TO_FAMILY.items() if fam == family]
    print(f"Organes: {', '.join(organs)}\n")

    # ========== ÉTAPE 1: Collecter toutes les images sources ==========
    all_source_images = []
    all_source_masks = []
    all_source_ids = []
    all_fold_ids = []

    for fold in folds:
        fold_dir = pannuke_dir / f"fold{fold}"
        images_path = fold_dir / "images.npy"
        masks_path = fold_dir / "masks.npy"
        types_path = fold_dir / "types.npy"

        if not images_path.exists():
            print(f"⚠️  Fold {fold}: fichiers manquants, skip")
            continue

        print(f"📂 Fold {fold}: Chargement...")

        images = np.load(images_path, mmap_mode='r')
        masks = np.load(masks_path, mmap_mode='r')
        types = np.load(types_path)

        for i in range(len(images)):
            organ_name = types[i].decode('utf-8') if isinstance(types[i], bytes) else types[i]
            if organ_name not in organs:
                continue

            # Charger en mémoire uniquement les images de cette famille
            image = np.array(images[i], dtype=np.uint8)
            mask = np.array(masks[i])

            all_source_images.append(image)
            all_source_masks.append(mask)
            # IMPORTANT: Source ID globalement unique (fold * 10000 + local_index)
            # Évite collision si même index local dans différents folds
            global_source_id = fold * 10000 + i
            all_source_ids.append(global_source_id)
            all_fold_ids.append(fold)

    n_total = len(all_source_images)
    print(f"✅ Total images sources collectées: {n_total}\n")

    # ========== ÉTAPE 2: Split train/val par source images ==========
    np.random.seed(seed)
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    n_train = int(train_ratio * n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    print(f"📊 Split train/val:")
    print(f"  Train: {len(train_indices)} images sources ({100*len(train_indices)/n_total:.1f}%)")
    print(f"  Val:   {len(val_indices)} images sources ({100*len(val_indices)/n_total:.1f}%)\n")

    # Pré-calculer NP et NT pour toutes les images sources (une seule fois)
    print("📝 Pré-calcul des targets NP/NT...")
    all_np_targets = []
    all_nt_targets = []
    for mask in tqdm(all_source_masks, desc="  Pré-calcul"):
        np_target = compute_np_target(mask)
        nt_target = compute_nt_target(mask)
        all_np_targets.append(np_target)
        all_nt_targets.append(nt_target)

    # ========== ÉTAPE 3: Traiter train et val séparément ==========
    global_stats = {
        'train': {'crops_kept': 0, 'crops_filtered': 0, 'layers_used': []},
        'val': {'crops_kept': 0, 'crops_filtered': 0, 'layers_used': []},
    }

    for split_name, split_indices in [('train', train_indices), ('val', val_indices)]:
        print(f"\n{'='*70}")
        print(f"Traitement split: {split_name.upper()}")
        print(f"{'='*70}\n")

        crops_data = {
            'images': [],
            'np_targets': [],
            'hv_targets': [],
            'nt_targets': [],
            'inst_maps': [],
            'weight_maps': [],
            'source_image_ids': [],
            'crop_positions': [],
            'fold_ids': [],
            'rotations': [],
        }

        split_stats = {
            'crops_kept': 0,
            'crops_filtered': 0,
            'layers_used': [],
        }

        # Traiter couche par couche jusqu'à max_samples
        print(f"📊 Génération par couche (max: {max_samples})...")

        for layer_idx, pos_name in enumerate(LAYER_ORDER):
            # Vérifier si on a atteint max_samples
            if split_stats['crops_kept'] >= max_samples:
                print(f"  ✅ Max samples atteint ({max_samples}), arrêt.")
                break

            x1, y1, x2, y2 = CROP_POSITIONS[pos_name]
            rotation = CROP_ROTATION_MAPPING[pos_name]

            print(f"\n  📦 Couche {layer_idx + 1}: {pos_name.upper()} (rotation: {rotation})")

            layer_kept = 0
            layer_filtered = 0

            # Traiter toutes les images sources DE CE SPLIT pour cette couche
            for idx in split_indices:
                # Vérifier si on a atteint max_samples
                if split_stats['crops_kept'] >= max_samples:
                    break

                image = all_source_images[idx]
                np_target = all_np_targets[idx]
                nt_target = all_nt_targets[idx]
                source_id = all_source_ids[idx]
                fold_id = all_fold_ids[idx]

                # ÉTAPE 1: Extraire crop RAW
                crop_raw = extract_raw_crop(
                    image, np_target, nt_target,
                    x1, y1, x2, y2
                )

                # ÉTAPE 2: Appliquer rotation
                img_rot, np_rot, nt_rot = apply_simple_rotation(
                    crop_raw['image'],
                    crop_raw['np_target'],
                    crop_raw['nt_target'],
                    rotation
                )

                # ÉTAPE 3: Finaliser APRÈS rotation
                hv_rot, inst_rot, weight_rot, n_instances, is_valid = finalize_crop_after_rotation(
                    img_rot, np_rot, nt_rot
                )

                # Filtrer si GT vide
                if not is_valid:
                    layer_filtered += 1
                    split_stats['crops_filtered'] += 1
                    continue

                crops_data['images'].append(img_rot)
                crops_data['np_targets'].append(np_rot)
                crops_data['hv_targets'].append(hv_rot)
                crops_data['nt_targets'].append(nt_rot)
                crops_data['inst_maps'].append(inst_rot)
                crops_data['weight_maps'].append(weight_rot)
                crops_data['source_image_ids'].append(source_id)
                crops_data['crop_positions'].append(pos_name)
                crops_data['fold_ids'].append(fold_id)
                crops_data['rotations'].append(rotation)

                layer_kept += 1
                split_stats['crops_kept'] += 1

            split_stats['layers_used'].append(pos_name)
            print(f"     Conservés: {layer_kept}, Filtrés: {layer_filtered}")
            print(f"     Total cumulé: {split_stats['crops_kept']}/{max_samples}")

        # Sauvegarde du split
        print(f"\n📊 Statistiques {split_name}:")
        print(f"  Crops conservés:   {split_stats['crops_kept']}")
        print(f"  Crops filtrés:     {split_stats['crops_filtered']}")
        print(f"  Couches utilisées: {len(split_stats['layers_used'])} ({', '.join(split_stats['layers_used'])})")

        if split_stats['crops_kept'] == 0:
            print(f"❌ ERREUR: Aucun crop généré pour {split_name}!")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{family}_{split_name}_v13_smart_crops.npz"

        print(f"\n💾 Conversion en arrays...")
        images_array = np.stack(crops_data['images'], axis=0)
        np_targets_array = np.stack(crops_data['np_targets'], axis=0)
        hv_targets_array = np.stack(crops_data['hv_targets'], axis=0)
        nt_targets_array = np.stack(crops_data['nt_targets'], axis=0)
        inst_maps_array = np.stack(crops_data['inst_maps'], axis=0)
        weight_maps_array = np.stack(crops_data['weight_maps'], axis=0)
        source_ids_array = np.array(crops_data['source_image_ids'], dtype=np.int32)
        crop_positions_array = np.array(crops_data['crop_positions'])
        fold_ids_array = np.array(crops_data['fold_ids'], dtype=np.int32)
        rotations_array = np.array(crops_data['rotations'])

        print(f"💾 Sauvegarde: {output_file}")
        np.savez_compressed(
            output_file,
            images=images_array,
            np_targets=np_targets_array,
            hv_targets=hv_targets_array,
            nt_targets=nt_targets_array,
            inst_maps=inst_maps_array,
            weight_maps=weight_maps_array,
            source_image_ids=source_ids_array,
            crop_positions=crop_positions_array,
            fold_ids=fold_ids_array,
            rotations=rotations_array,
        )

        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"✅ Fichier créé: {file_size_mb:.1f} MB")

        # Mettre à jour stats globales
        global_stats[split_name] = split_stats

    # ========== RÉSUMÉ FINAL ==========
    print(f"\n{'='*70}")
    print(f"📊 RÉSUMÉ FINAL")
    print(f"{'='*70}")
    print(f"  Images sources:   {n_total}")
    print(f"  Train crops:      {global_stats['train']['crops_kept']}")
    print(f"  Val crops:        {global_stats['val']['crops_kept']}")
    print(f"  Total crops:      {global_stats['train']['crops_kept'] + global_stats['val']['crops_kept']}")
    print(f"\n✅ GÉNÉRATION COMPLÈTE - Train et Val sauvegardés séparément")

    return global_stats


def main():
    parser = argparse.ArgumentParser(
        description="Génération V13 Smart Crops avec algorithme par couche"
    )
    parser.add_argument(
        '--pannuke_dir',
        type=Path,
        default=Path('/home/amar/data/PanNuke'),
        help="Répertoire PanNuke"
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('data/family_data_v13_smart_crops'),
        help="Répertoire de sortie"
    )
    parser.add_argument(
        '--family',
        type=str,
        required=True,
        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'],
        help="Famille tissulaire"
    )
    parser.add_argument(
        '--folds',
        type=int,
        nargs='+',
        default=[0, 1, 2],
        help="Folds à traiter (défaut: 0 1 2)"
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=5000,
        help="Nombre maximum de samples à générer (défaut: 5000)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Seed pour reproductibilité (défaut: 42)"
    )

    args = parser.parse_args()

    # Validation
    if not args.pannuke_dir.exists():
        print(f"❌ ERREUR: PanNuke directory non trouvé: {args.pannuke_dir}")
        sys.exit(1)

    # Génération
    stats = generate_smart_crops_from_pannuke(
        pannuke_dir=args.pannuke_dir,
        output_dir=args.output_dir,
        family=args.family,
        folds=args.folds,
        max_samples=args.max_samples,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
