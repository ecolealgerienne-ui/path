#!/usr/bin/env python3
"""
Sauvegarde quelques échantillons PanNuke en PNG pour tester l'IHM Gradio.

Supporte deux modes:
- 256×256 (raw): Pour tester l'InputRouter (preprocessing automatique)
- 224×224 (cropped): Pour tester le pipeline direct

Usage:
    # Mode par défaut: sauve les deux formats (256 et 224)
    python scripts/utils/save_pannuke_samples.py --data_dir /chemin/vers/PanNuke

    # Mode raw seulement (256×256 pour tester InputRouter)
    python scripts/utils/save_pannuke_samples.py --data_dir /chemin/vers/PanNuke --raw_only

    # Mode cropped seulement (224×224)
    python scripts/utils/save_pannuke_samples.py --data_dir /chemin/vers/PanNuke --crop_only

    # Depuis un fichier NPZ directement
    python scripts/utils/save_pannuke_samples.py --npz_file /chemin/vers/data.npz
"""

import numpy as np
from pathlib import Path
import cv2
import random
import argparse

# Configuration par défaut
DEFAULT_DATA_DIR = Path("/home/amar/data/PanNuke")
OUTPUT_DIR = Path("data/samples")
N_IMAGES_PER_ORGAN = 5
ORGANS = ["Prostate", "Breast", "Colon"]  # 3 organes différents

# Crop centre 224×224 depuis image 256×256
# Mêmes coordonnées que prepare_v13_smart_crops.py
CROP_CENTER = (16, 16, 240, 240)  # (x1, y1, x2, y2)
PANNUKE_SIZE = 256
OUTPUT_SIZE = 224


def load_pannuke_data(data_dir: Path = None, npz_file: Path = None):
    """
    Charge les données PanNuke depuis un dossier ou fichier NPZ.

    Returns:
        (images, types) ou (images, None) si pas de types
    """
    if npz_file and Path(npz_file).exists():
        print(f"Chargement depuis NPZ: {npz_file}")
        data = np.load(npz_file, allow_pickle=True)

        # Chercher la clé images
        if 'images' in data:
            images = data['images']
        elif 'image' in data:
            images = data['image']
        else:
            # Prendre la première clé
            keys = list(data.keys())
            print(f"  Clés disponibles: {keys}")
            images = data[keys[0]]

        # Chercher les types si disponibles
        types = None
        if 'types' in data:
            types = data['types']
        elif 'type' in data:
            types = data['type']

        return images, types

    # Sinon chercher dans le dossier
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    print(f"Chargement des données depuis {data_dir}...")

    # Chercher les fichiers (différentes conventions de nommage)
    possible_paths = [
        data_dir / "fold0" / "images.npy",
        data_dir / "fold1" / "images.npy",
        data_dir / "fold2" / "images.npy",
        data_dir / "fold_0" / "images.npy",
        data_dir / "images.npy",
    ]

    for p in possible_paths:
        if p.exists():
            images = np.load(p)
            types_path = p.parent / "types.npy"
            types = np.load(types_path) if types_path.exists() else None
            return images, types

    raise FileNotFoundError(
        f"Données PanNuke non trouvées dans {data_dir}\n"
        f"Chemins testés: {possible_paths}"
    )


def main():
    parser = argparse.ArgumentParser(description="Sauvegarder des échantillons PanNuke")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Chemin vers le dossier PanNuke")
    parser.add_argument("--npz_file", type=str, default=None,
                        help="Chemin vers un fichier NPZ directement")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="Dossier de sortie pour les images PNG")
    parser.add_argument("--n_per_organ", type=int, default=N_IMAGES_PER_ORGAN,
                        help="Nombre d'images par organe (ou total si pas d'organes)")
    parser.add_argument("--n_total", type=int, default=None,
                        help="Nombre total d'images (ignore --organs)")
    parser.add_argument("--organs", nargs="+", default=ORGANS,
                        help="Liste des organes à extraire")
    parser.add_argument("--raw_only", action="store_true",
                        help="Sauvegarder uniquement les images 256×256 (InputRouter)")
    parser.add_argument("--crop_only", action="store_true",
                        help="Sauvegarder uniquement les images 224×224 (direct)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed pour reproductibilité")
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Déterminer les modes de sauvegarde
    save_raw = not args.crop_only  # 256×256
    save_crop = not args.raw_only  # 224×224

    if args.raw_only and args.crop_only:
        print("❌ --raw_only et --crop_only sont mutuellement exclusifs")
        return

    # Charger les données
    try:
        data_dir = Path(args.data_dir) if args.data_dir else None
        npz_file = Path(args.npz_file) if args.npz_file else None
        images, types = load_pannuke_data(data_dir, npz_file)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    print(f"Images chargées: {images.shape}")
    if types is not None:
        print(f"Organes disponibles: {np.unique(types)}")

    saved_count = 0

    # Mode 1: Par organe (si types disponibles et pas --n_total)
    if types is not None and args.n_total is None:
        for organ in args.organs:
            organ_indices = np.where(types == organ)[0]

            if len(organ_indices) == 0:
                print(f"⚠️ Aucune image trouvée pour {organ}")
                continue

            print(f"\n{organ}: {len(organ_indices)} images disponibles")

            n_select = min(args.n_per_organ, len(organ_indices))
            selected = random.sample(list(organ_indices), n_select)

            for i, idx in enumerate(selected):
                saved_count += save_image(
                    images[idx], output_dir, f"{organ.lower()}_{i+1:02d}",
                    save_raw, save_crop
                )

    # Mode 2: Sélection aléatoire globale
    else:
        n_total = args.n_total if args.n_total else args.n_per_organ * len(args.organs)
        n_select = min(n_total, len(images))
        selected = random.sample(range(len(images)), n_select)

        print(f"\nSélection aléatoire de {n_select} images...")

        for i, idx in enumerate(selected):
            saved_count += save_image(
                images[idx], output_dir, f"sample_{i+1:03d}",
                save_raw, save_crop
            )

    # Résumé
    print(f"\n{'='*60}")
    print(f"✅ {saved_count} fichiers sauvegardés dans {output_dir}/")

    if save_raw:
        print(f"\n📦 Images 256×256 (raw):")
        print(f"   → Pour tester l'InputRouter (preprocessing automatique)")

    if save_crop:
        print(f"\n📦 Images 224×224 (cropped):")
        print(f"   → Pour tester le pipeline direct")

    print(f"\nPour tester dans Gradio:")
    print(f"  1. Lancer l'interface:")
    print(f"     python -m src.ui.app --preload")
    print(f"  2. Aller sur http://localhost:7860")
    print(f"  3. Uploader les images depuis {output_dir}/")


def save_image(img: np.ndarray, output_dir: Path, name: str,
               save_raw: bool, save_crop: bool) -> int:
    """
    Sauvegarde une image en format raw (256) et/ou crop (224).

    Returns:
        Nombre de fichiers sauvegardés
    """
    count = 0

    # Convertir de [0,1] à [0,255] si nécessaire
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # Vérifier la taille
    h, w = img.shape[:2]

    # Sauvegarder 256×256 (raw)
    if save_raw:
        if h == PANNUKE_SIZE and w == PANNUKE_SIZE:
            filename = f"{name}_256.png"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"  ✓ {filename} (256×256 raw → InputRouter)")
            count += 1
        else:
            print(f"  ⚠️ Image {h}×{w} ignorée pour raw (attendu 256×256)")

    # Sauvegarder 224×224 (crop)
    if save_crop:
        if h == PANNUKE_SIZE and w == PANNUKE_SIZE:
            x1, y1, x2, y2 = CROP_CENTER
            img_cropped = img[y1:y2, x1:x2]
        elif h == OUTPUT_SIZE and w == OUTPUT_SIZE:
            img_cropped = img
        else:
            # Resize si autre taille
            img_cropped = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE))

        filename = f"{name}_224.png"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR))
        print(f"  ✓ {filename} (224×224 direct)")
        count += 1

    return count


if __name__ == "__main__":
    main()
