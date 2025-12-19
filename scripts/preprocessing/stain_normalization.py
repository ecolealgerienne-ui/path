#!/usr/bin/env python3
"""
Normalisation de coloration Macenko pour images H&E.

Référence: Macenko et al., "A method for normalizing histology slides
for quantitative analysis", ISBI 2009.

Usage:
    python scripts/preprocessing/stain_normalization.py --input image.png --output normalized.png
    python scripts/preprocessing/stain_normalization.py --input-dir data/raw --output-dir data/normalized
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import cv2


class MacenkoNormalizer:
    """
    Normaliseur de coloration Macenko.

    Décompose l'image en composantes H&E puis reconstruit
    avec des vecteurs de stain de référence.
    """

    # Vecteurs de stain de référence (H&E standard)
    HE_REF = np.array([
        [0.5626, 0.2159],  # Hématoxyline
        [0.7201, 0.8012],  # Éosine
        [0.4062, 0.5581],  # Résiduel
    ])

    MAX_C_REF = np.array([1.9705, 1.0308])

    def __init__(self, alpha: float = 1.0, beta: float = 0.15):
        """
        Args:
            alpha: Percentile pour seuillage (1 = 1%, 99 = 99%)
            beta: Seuil de luminosité pour filtrer le fond
        """
        self.alpha = alpha
        self.beta = beta

    def _get_stain_matrix(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extrait la matrice de stain de l'image."""
        # Convertir en OD (Optical Density)
        img = img.astype(np.float64) + 1
        od = -np.log(img / 255.0)

        # Filtrer le fond (faible OD)
        od_flat = od.reshape(-1, 3)
        od_threshold = od_flat[np.all(od_flat > self.beta, axis=1)]

        if len(od_threshold) < 10:
            return self.HE_REF, self.MAX_C_REF

        # SVD pour trouver les directions principales
        _, _, V = np.linalg.svd(od_threshold, full_matrices=False)

        # Projeter sur le plan des 2 premiers composants
        plane = V[:2, :]
        proj = np.dot(od_threshold, plane.T)

        # Trouver les angles extrêmes
        angles = np.arctan2(proj[:, 1], proj[:, 0])

        min_angle = np.percentile(angles, self.alpha)
        max_angle = np.percentile(angles, 100 - self.alpha)

        # Vecteurs de stain
        v1 = np.array([np.cos(min_angle), np.sin(min_angle)])
        v2 = np.array([np.cos(max_angle), np.sin(max_angle)])

        # Reconstruire en 3D
        stain1 = np.dot(v1, plane)
        stain2 = np.dot(v2, plane)

        # Normaliser
        stain1 /= np.linalg.norm(stain1)
        stain2 /= np.linalg.norm(stain2)

        # Assurer que H est plus foncé que E
        if stain1[0] < stain2[0]:
            stain1, stain2 = stain2, stain1

        stain_matrix = np.array([stain1, stain2]).T

        # Calculer les concentrations max
        od_flat_valid = od_flat[np.all(od_flat > self.beta, axis=1)]
        concentrations = np.linalg.lstsq(stain_matrix, od_flat_valid.T, rcond=None)[0]
        max_c = np.percentile(concentrations, 99, axis=1)

        return stain_matrix, max_c

    def normalize(self, img: np.ndarray, target_stain: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalise une image H&E.

        Args:
            img: Image RGB (H, W, 3) uint8
            target_stain: Matrice de stain cible (optionnel)

        Returns:
            Image normalisée (H, W, 3) uint8
        """
        if target_stain is None:
            target_stain = self.HE_REF[:, :2]
            target_max_c = self.MAX_C_REF
        else:
            target_max_c = self.MAX_C_REF

        # Extraire stain source
        source_stain, source_max_c = self._get_stain_matrix(img)

        # Convertir en OD
        img_float = img.astype(np.float64) + 1
        od = -np.log(img_float / 255.0)
        od_flat = od.reshape(-1, 3)

        # Déconvolution
        try:
            concentrations = np.linalg.lstsq(source_stain, od_flat.T, rcond=None)[0]
        except np.linalg.LinAlgError:
            return img

        # Normaliser les concentrations
        concentrations = concentrations * (target_max_c / source_max_c)[:, np.newaxis]

        # Reconstruire avec stain cible
        od_normalized = np.dot(target_stain, concentrations).T
        od_normalized = od_normalized.reshape(od.shape)

        # Convertir en RGB
        img_normalized = 255 * np.exp(-od_normalized) - 1
        img_normalized = np.clip(img_normalized, 0, 255).astype(np.uint8)

        return img_normalized


def normalize_image(
    input_path: Path,
    output_path: Path,
    normalizer: MacenkoNormalizer
) -> bool:
    """Normalise une image."""
    try:
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"Erreur lecture: {input_path}")
            return False

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        normalized = normalizer.normalize(img_rgb)
        normalized_bgr = cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), normalized_bgr)
        return True
    except Exception as e:
        print(f"Erreur: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Normalisation de coloration Macenko"
    )
    parser.add_argument("--input", type=str, help="Image d'entrée")
    parser.add_argument("--output", type=str, help="Image de sortie")
    parser.add_argument("--input-dir", type=str, help="Répertoire d'entrée")
    parser.add_argument("--output-dir", type=str, help="Répertoire de sortie")
    parser.add_argument("--alpha", type=float, default=1.0, help="Percentile seuil")
    parser.add_argument("--beta", type=float, default=0.15, help="Seuil luminosité")

    args = parser.parse_args()

    normalizer = MacenkoNormalizer(alpha=args.alpha, beta=args.beta)

    if args.input and args.output:
        success = normalize_image(Path(args.input), Path(args.output), normalizer)
        if success:
            print(f"✓ Normalisé: {args.output}")
    elif args.input_dir and args.output_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)

        extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        images = [f for f in input_dir.rglob("*") if f.suffix.lower() in extensions]

        print(f"Normalisation de {len(images)} images...")

        success_count = 0
        for img_path in images:
            rel_path = img_path.relative_to(input_dir)
            out_path = output_dir / rel_path

            if normalize_image(img_path, out_path, normalizer):
                success_count += 1

        print(f"\n✅ {success_count}/{len(images)} images normalisées")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
