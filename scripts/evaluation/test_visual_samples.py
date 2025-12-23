#!/usr/bin/env python3
"""
Script de test visuel sur quelques échantillons représentatifs.

Génère des visualisations comparant prédictions vs ground truth pour chaque famille.

Usage:
    python scripts/evaluation/test_visual_samples.py \\
        --data_dir /path/to/PanNuke \\
        --checkpoints_dir models/checkpoints \\
        --output_dir results/visual_test \\
        --n_per_family 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.organ_head import OrganHead
from src.models.loader import ModelLoader
from src.preprocessing import preprocess_image

ORGAN_TO_FAMILY = {
    "Breast": "glandular", "Prostate": "glandular", "Thyroid": "glandular",
    "Pancreatic": "glandular", "Adrenal_gland": "glandular",
    "Colon": "digestive", "Stomach": "digestive", "Esophagus": "digestive", "Bile-duct": "digestive",
    "Kidney": "urologic", "Bladder": "urologic", "Testis": "urologic",
    "Ovarian": "urologic", "Uterus": "urologic", "Cervix": "urologic",
    "Lung": "respiratory", "Liver": "respiratory",
    "Skin": "epidermal", "HeadNeck": "epidermal",
}

FAMILIES = ["glandular", "digestive", "urologic", "respiratory", "epidermal"]


def load_pannuke_fold(data_dir: Path, fold: int):
    """Charge un fold PanNuke."""
    fold_dir = data_dir / f"Fold {fold}"
    images = np.load(fold_dir / "images.npy", mmap_mode='r')
    masks = np.load(fold_dir / "masks.npy", mmap_mode='r')
    types = np.load(fold_dir / "types.npy")
    return images, masks, types


def visualize_sample(image, mask, np_pred, organ_name, family_name, output_path):
    """Crée une visualisation comparative."""
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig)

    # Image originale
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title(f"Image H&E\n{organ_name} ({family_name})", fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Ground truth
    ax2 = fig.add_subplot(gs[0, 1])
    np_gt = mask[:, :, 1:].sum(axis=-1) > 0
    ax2.imshow(image, alpha=0.5)
    ax2.imshow(np_gt, alpha=0.5, cmap='Reds')
    ax2.set_title("Ground Truth\n(Union des 5 types)", fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Prédiction
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(image, alpha=0.5)
    # Resize pred 224 → 256 pour visualisation
    import torch.nn.functional as F
    np_pred_t = torch.from_numpy(np_pred).unsqueeze(0).unsqueeze(0)
    np_pred_256 = F.interpolate(np_pred_t, size=(256, 256), mode='bilinear', align_corners=False).squeeze().numpy()
    ax3.imshow(np_pred_256, alpha=0.5, cmap='Greens')
    ax3.set_title("Prédiction HoVer-Net\n(Nuclear Presence)", fontsize=12, fontweight='bold')
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Sauvegardé: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Test visuel sur échantillons représentatifs")
    parser.add_argument("--data_dir", type=str, required=True, help="Répertoire PanNuke")
    parser.add_argument("--checkpoints_dir", type=str, default="models/checkpoints", help="Répertoire checkpoints")
    parser.add_argument("--output_dir", type=str, default="results/visual_test", help="Répertoire de sortie")
    parser.add_argument("--fold", type=int, default=2, help="Fold PanNuke (0, 1, 2)")
    parser.add_argument("--n_per_family", type=int, default=3, help="Nombre d'échantillons par famille")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("="*80)
    print("TEST VISUEL OPTIMUSGATE MULTI-FAMILLE")
    print("="*80)

    # Charger modèles
    print("\n📥 Chargement des modèles...")
    backbone = ModelLoader.load_hoptimus0(device=str(device))

    family_models = {}
    for family in FAMILIES:
        checkpoint_path = checkpoints_dir / f"hovernet_{family}_best.pth"
        model = HoVerNetDecoder(embed_dim=1536, n_classes=5).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        family_models[family] = model

    print("✅ Modèles chargés")

    # Charger données
    print(f"\n📥 Chargement Fold {args.fold}...")
    images, masks, types = load_pannuke_fold(data_dir, args.fold)

    # Grouper par famille
    family_indices = {f: [] for f in FAMILIES}
    for idx, organ_name in enumerate(types):
        family = ORGAN_TO_FAMILY[organ_name]
        family_indices[family].append((idx, organ_name))

    # Sélectionner échantillons aléatoires par famille
    np.random.seed(42)

    print("\n" + "="*80)
    print("GÉNÉRATION VISUALISATIONS")
    print("="*80)

    for family in FAMILIES:
        print(f"\n📊 Famille {family.upper()}")

        if len(family_indices[family]) == 0:
            print("  ⚠️  Aucun échantillon disponible")
            continue

        # Sélectionner n_per_family échantillons
        n_samples = min(args.n_per_family, len(family_indices[family]))
        selected = np.random.choice(len(family_indices[family]), n_samples, replace=False)

        hovernet = family_models[family]

        for i, sample_idx in enumerate(selected):
            idx, organ_name = family_indices[family][sample_idx]

            image = images[idx]
            mask = masks[idx]

            # Prétraitement
            if image.dtype != np.uint8:
                image = image.clip(0, 255).astype(np.uint8)

            tensor = preprocess_image(image, device=str(device))

            # Extraction features
            with torch.no_grad():
                features = backbone.forward_features(tensor)
                patch_tokens = features[:, 1:257, :]

                # Prédiction
                np_out, hv_out, nt_out = hovernet(patch_tokens)

                # Convertir en probabilité
                np_pred = torch.softmax(np_out, dim=1)[0, 1].cpu().numpy()  # Canal nuclei (224, 224)

            # Visualisation
            output_path = output_dir / f"{family}_{i+1}_{organ_name}.png"
            visualize_sample(image, mask, np_pred, organ_name, family, output_path)

    print("\n" + "="*80)
    print(f"✅ Visualisations sauvegardées: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
