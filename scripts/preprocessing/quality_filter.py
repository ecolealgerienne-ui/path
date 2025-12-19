#!/usr/bin/env python3
"""
Filtrage qualité des images histopathologiques.

Détecte et filtre:
- Images floues
- Artefacts (bulles, plis, marquages)
- Fond blanc excessif
- Problèmes de coloration

Usage:
    python scripts/preprocessing/quality_filter.py --input-dir tiles/ --output-dir filtered/
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import cv2


class QualityFilter:
    """Filtre de qualité pour images histopathologiques."""

    def __init__(
        self,
        blur_threshold: float = 100.0,
        tissue_min: float = 0.3,
        saturation_min: float = 0.1,
        brightness_range: Tuple[int, int] = (30, 220),
    ):
        """
        Args:
            blur_threshold: Seuil de Laplacian variance (< = flou)
            tissue_min: Proportion minimale de tissu
            saturation_min: Saturation minimale moyenne
            brightness_range: Plage de luminosité acceptable
        """
        self.blur_threshold = blur_threshold
        self.tissue_min = tissue_min
        self.saturation_min = saturation_min
        self.brightness_range = brightness_range

    def check_blur(self, image: np.ndarray) -> Tuple[bool, float]:
        """Vérifie le flou via variance du Laplacien."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_sharp = laplacian_var >= self.blur_threshold

        return is_sharp, laplacian_var

    def check_tissue_content(self, image: np.ndarray) -> Tuple[bool, float]:
        """Vérifie la proportion de tissu."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Pixels non-blancs = tissu
        tissue_mask = gray < 220
        tissue_ratio = np.mean(tissue_mask)

        has_tissue = tissue_ratio >= self.tissue_min

        return has_tissue, tissue_ratio

    def check_saturation(self, image: np.ndarray) -> Tuple[bool, float]:
        """Vérifie la saturation (coloration H&E)."""
        if len(image.shape) != 3:
            return False, 0.0

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].mean() / 255.0

        has_color = saturation >= self.saturation_min

        return has_color, saturation

    def check_brightness(self, image: np.ndarray) -> Tuple[bool, float]:
        """Vérifie la luminosité."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        brightness = gray.mean()

        in_range = self.brightness_range[0] <= brightness <= self.brightness_range[1]

        return in_range, brightness

    def detect_artifacts(self, image: np.ndarray) -> Tuple[bool, Dict]:
        """Détecte les artefacts courants."""
        artifacts = {}

        if len(image.shape) != 3:
            return True, artifacts

        # Détection de bulles (zones très claires circulaires)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        white_ratio = np.mean(white_mask) / 255

        if white_ratio > 0.3:
            artifacts['bubbles'] = white_ratio

        # Détection de plis (lignes sombres)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        if lines is not None and len(lines) > 20:
            artifacts['folds'] = len(lines)

        # Détection de marquages (couleurs non-H&E)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Bleu marqueur
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        if np.mean(blue_mask) > 0.05:
            artifacts['blue_marker'] = np.mean(blue_mask)

        # Noir marqueur
        black_mask = gray < 20
        if np.mean(black_mask) > 0.1:
            artifacts['black_marker'] = np.mean(black_mask)

        no_artifacts = len(artifacts) == 0

        return no_artifacts, artifacts

    def evaluate(self, image: np.ndarray) -> Dict:
        """Évalue la qualité complète d'une image."""
        is_sharp, blur_score = self.check_blur(image)
        has_tissue, tissue_ratio = self.check_tissue_content(image)
        has_color, saturation = self.check_saturation(image)
        in_range, brightness = self.check_brightness(image)
        no_artifacts, artifacts = self.detect_artifacts(image)

        passed = all([is_sharp, has_tissue, has_color, in_range, no_artifacts])

        return {
            'passed': passed,
            'blur': {'passed': is_sharp, 'score': blur_score},
            'tissue': {'passed': has_tissue, 'ratio': tissue_ratio},
            'saturation': {'passed': has_color, 'value': saturation},
            'brightness': {'passed': in_range, 'value': brightness},
            'artifacts': {'passed': no_artifacts, 'detected': artifacts},
        }


def filter_directory(
    input_dir: Path,
    output_dir: Path,
    qfilter: QualityFilter,
    move_rejected: bool = False
) -> Dict:
    """Filtre un répertoire d'images."""
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    images = [f for f in input_dir.rglob('*') if f.suffix.lower() in extensions]

    output_dir.mkdir(parents=True, exist_ok=True)

    if move_rejected:
        rejected_dir = output_dir.parent / 'rejected'
        rejected_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'total': len(images),
        'passed': 0,
        'rejected': 0,
        'rejection_reasons': {
            'blur': 0,
            'tissue': 0,
            'saturation': 0,
            'brightness': 0,
            'artifacts': 0,
        }
    }

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = qfilter.evaluate(img_rgb)

        rel_path = img_path.relative_to(input_dir)

        if result['passed']:
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), img)
            stats['passed'] += 1
        else:
            stats['rejected'] += 1

            # Compter les raisons
            for key in ['blur', 'tissue', 'saturation', 'brightness', 'artifacts']:
                if not result[key]['passed']:
                    stats['rejection_reasons'][key] += 1

            if move_rejected:
                out_path = rejected_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), img)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Filtrage qualité des images histopathologiques"
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Répertoire d'entrée")
    parser.add_argument("--output-dir", type=str, required=True, help="Répertoire de sortie")
    parser.add_argument("--blur-threshold", type=float, default=100.0, help="Seuil flou")
    parser.add_argument("--tissue-min", type=float, default=0.3, help="Tissu minimum")
    parser.add_argument("--move-rejected", action="store_true", help="Déplacer les rejetées")

    args = parser.parse_args()

    qfilter = QualityFilter(
        blur_threshold=args.blur_threshold,
        tissue_min=args.tissue_min,
    )

    print(f"Filtrage de {args.input_dir}...")

    stats = filter_directory(
        Path(args.input_dir),
        Path(args.output_dir),
        qfilter,
        args.move_rejected
    )

    print(f"\n{'='*40}")
    print(f"Résultats:")
    print(f"  Total: {stats['total']}")
    print(f"  Acceptées: {stats['passed']} ({stats['passed']/max(1,stats['total'])*100:.1f}%)")
    print(f"  Rejetées: {stats['rejected']}")
    print(f"\nRaisons de rejet:")
    for reason, count in stats['rejection_reasons'].items():
        if count > 0:
            print(f"  - {reason}: {count}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
