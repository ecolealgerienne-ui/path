#!/usr/bin/env python3
"""
Download evaluation datasets for Ground Truth comparison.

Supports:
- PanNuke (7,901 images, 5 classes, 19 organs)
- CoNSeP (41 images, colorectal adenocarcinoma)
- MoNuSAC (209 images, 4 immune cell types)
- Lizard (291 images, 500k+ nuclei)

Usage:
    # Download all datasets
    python scripts/evaluation/download_evaluation_datasets.py --dataset all

    # Download specific dataset
    python scripts/evaluation/download_evaluation_datasets.py --dataset consep

    # Specify output directory
    python scripts/evaluation/download_evaluation_datasets.py --dataset pannuke --output_dir data/evaluation
"""

import argparse
import os
import sys
import zipfile
import shutil
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
from tqdm import tqdm

# Dataset configurations
DATASETS = {
    "pannuke": {
        "url": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/",
        "format": "npy",
        "classes": 6,  # Including background
        "description": "PanNuke - 7,901 images, 5 classes, 19 organs",
        "size": "~1.5 GB (compressed)",
        "files": {
            "fold1": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_1.zip",
            "fold2": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip",
            "fold3": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip",
        }
    },
    "consep": {
        "url": "https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/",
        "format": "mat",
        "classes": 7,  # Needs mapping to 5
        "description": "CoNSeP - 41 images, colorectal adenocarcinoma",
        "size": "~70 MB",
        "files": {
            # Direct download URLs (may require manual download)
            "dataset": [
                "https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep_dataset.zip",
                # Backup: Google Drive link (from HoVer-Net paper)
                # Manual download required
            ]
        },
        "manual_instructions": """
CoNSeP dataset download may require manual steps:

1. Visit: https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/
2. Download "consep_dataset.zip" (70 MB)
3. Place in: data/evaluation/consep/consep_dataset.zip
4. Re-run this script or extract manually:
   unzip data/evaluation/consep/consep_dataset.zip -d data/evaluation/consep/
        """
    },
    "monusac": {
        "url": "https://huggingface.co/datasets/RationAI/MoNuSAC",
        "format": "huggingface",
        "classes": 4,  # Needs mapping to 5
        "description": "MoNuSAC - 209 images, 4 immune cell types",
        "size": "~500 MB",
        "files": {}  # Requires huggingface_hub
    },
    "lizard": {
        "url": "https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/",
        "format": "npy",
        "classes": 6,
        "description": "Lizard - 291 images, 500k+ nuclei (colon)",
        "size": "~2 GB",
        "files": {
            "dataset": "https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/lizard_images_and_labels.zip"
        }
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: str = "Downloading") -> None:
    """Download a file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urlretrieve(url, str(output_path), reporthook=t.update_to)


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extracted to {extract_to}")


def download_pannuke(output_dir: Path, folds: Optional[list] = None) -> None:
    """Download PanNuke dataset."""
    print("\n" + "="*70)
    print("📦 DOWNLOADING PANNUKE")
    print("="*70)

    if folds is None:
        folds = [1, 2, 3]  # All folds by default

    pannuke_dir = output_dir / "pannuke"
    pannuke_dir.mkdir(parents=True, exist_ok=True)

    config = DATASETS["pannuke"]

    for fold_num in folds:
        fold_key = f"fold{fold_num}"
        if fold_key not in config["files"]:
            print(f"⚠️ Warning: Fold {fold_num} not available")
            continue

        url = config["files"][fold_key]
        zip_path = pannuke_dir / f"fold_{fold_num}.zip"

        # Download
        if zip_path.exists():
            print(f"✅ Fold {fold_num} already downloaded: {zip_path}")
        else:
            print(f"\nDownloading Fold {fold_num}...")
            download_file(url, zip_path, desc=f"Fold {fold_num}")

        # Extract
        fold_dir = pannuke_dir / f"Fold {fold_num}"
        if fold_dir.exists():
            print(f"✅ Fold {fold_num} already extracted")
        else:
            extract_zip(zip_path, pannuke_dir)

    print(f"\n✅ PanNuke downloaded to: {pannuke_dir}")


def download_consep(output_dir: Path) -> None:
    """Download CoNSeP dataset."""
    print("\n" + "="*70)
    print("📦 DOWNLOADING CONSEP")
    print("="*70)

    consep_dir = output_dir / "consep"
    consep_dir.mkdir(parents=True, exist_ok=True)

    config = DATASETS["consep"]
    urls = config["files"]["dataset"]
    if not isinstance(urls, list):
        urls = [urls]

    zip_path = consep_dir / "consep_dataset.zip"

    # Check if already extracted
    if (consep_dir / "Test").exists() or (consep_dir / "Train").exists():
        print(f"✅ CoNSeP already extracted")
        print(f"✅ CoNSeP available at: {consep_dir}")
        return

    # Check if zip exists
    if zip_path.exists() and zip_path.stat().st_size > 1_000_000:  # > 1 MB
        print(f"✅ CoNSeP already downloaded: {zip_path}")
        try:
            extract_zip(zip_path, consep_dir)
            print(f"\n✅ CoNSeP extracted to: {consep_dir}")
            return
        except Exception as e:
            print(f"⚠️ Error extracting: {e}")
            print("Trying to re-download...")
            zip_path.unlink()

    # Try each URL
    download_success = False
    for i, url in enumerate(urls):
        try:
            print(f"\nTrying URL {i+1}/{len(urls)}...")
            print(f"Downloading CoNSeP from: {url}")
            download_file(url, zip_path, desc="CoNSeP")

            # Verify it's a valid zip
            if zip_path.stat().st_size < 1_000_000:  # < 1 MB is suspicious
                print(f"⚠️ Downloaded file is too small ({zip_path.stat().st_size} bytes)")
                print("This may be an HTML redirect page, not the actual dataset.")
                zip_path.unlink()
                continue

            # Try to extract
            extract_zip(zip_path, consep_dir)
            download_success = True
            print(f"\n✅ CoNSeP downloaded and extracted to: {consep_dir}")
            break

        except Exception as e:
            print(f"⚠️ Failed with URL {i+1}: {e}")
            if zip_path.exists():
                zip_path.unlink()
            continue

    if not download_success:
        print("\n" + "="*70)
        print("❌ AUTOMATIC DOWNLOAD FAILED - MANUAL DOWNLOAD REQUIRED")
        print("="*70)
        print(config.get("manual_instructions", ""))
        print("\nAfter manual download, the directory should contain:")
        print("  data/evaluation/consep/Train/")
        print("  data/evaluation/consep/Test/")
        print("="*70)


def download_monusac(output_dir: Path) -> None:
    """Download MoNuSAC dataset from Hugging Face."""
    print("\n" + "="*70)
    print("📦 DOWNLOADING MONUSAC")
    print("="*70)

    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: 'datasets' library not found")
        print("Install with: pip install datasets")
        return

    monusac_dir = output_dir / "monusac"
    monusac_dir.mkdir(parents=True, exist_ok=True)

    print("\nDownloading from Hugging Face...")
    dataset = load_dataset("RationAI/MoNuSAC", cache_dir=str(monusac_dir))

    print(f"\n✅ MoNuSAC downloaded to: {monusac_dir}")
    print(f"Dataset info: {dataset}")


def download_lizard(output_dir: Path) -> None:
    """Download Lizard dataset."""
    print("\n" + "="*70)
    print("📦 DOWNLOADING LIZARD")
    print("="*70)

    lizard_dir = output_dir / "lizard"
    lizard_dir.mkdir(parents=True, exist_ok=True)

    config = DATASETS["lizard"]
    url = config["files"]["dataset"]
    zip_path = lizard_dir / "lizard_dataset.zip"

    # Download
    if zip_path.exists():
        print(f"✅ Lizard already downloaded: {zip_path}")
    else:
        print("\nDownloading Lizard...")
        download_file(url, zip_path, desc="Lizard")

    # Extract
    if (lizard_dir / "images").exists():
        print(f"✅ Lizard already extracted")
    else:
        extract_zip(zip_path, lizard_dir)

    print(f"\n✅ Lizard downloaded to: {lizard_dir}")


def show_dataset_info():
    """Display information about available datasets."""
    print("\n" + "="*70)
    print("AVAILABLE DATASETS FOR GROUND TRUTH EVALUATION")
    print("="*70 + "\n")

    for i, (name, config) in enumerate(DATASETS.items(), 1):
        priority_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📦"
        print(f"{priority_emoji} {name.upper()}")
        print(f"   Description: {config['description']}")
        print(f"   Format: {config['format']}")
        print(f"   Classes: {config['classes']}")
        print(f"   Size: {config['size']}")
        print(f"   URL: {config['url']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download evaluation datasets for Ground Truth comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show dataset information
  python scripts/evaluation/download_evaluation_datasets.py --info

  # Download all datasets
  python scripts/evaluation/download_evaluation_datasets.py --dataset all

  # Download PanNuke only (all folds)
  python scripts/evaluation/download_evaluation_datasets.py --dataset pannuke

  # Download specific PanNuke folds
  python scripts/evaluation/download_evaluation_datasets.py --dataset pannuke --folds 2 3

  # Download CoNSeP (fast, 70 MB)
  python scripts/evaluation/download_evaluation_datasets.py --dataset consep
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "pannuke", "consep", "monusac", "lizard"],
        help="Dataset to download"
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/evaluation"),
        help="Output directory (default: data/evaluation)"
    )

    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="PanNuke folds to download (default: all)"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show dataset information and exit"
    )

    args = parser.parse_args()

    if args.info:
        show_dataset_info()
        return

    if not args.dataset:
        parser.print_help()
        print("\n❌ Error: --dataset is required (or use --info)")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Output directory: {args.output_dir.absolute()}")

    # Download requested dataset(s)
    if args.dataset == "all":
        download_pannuke(args.output_dir, args.folds)
        download_consep(args.output_dir)
        try:
            download_monusac(args.output_dir)
        except Exception as e:
            print(f"⚠️ Warning: Could not download MoNuSAC: {e}")
        try:
            download_lizard(args.output_dir)
        except Exception as e:
            print(f"⚠️ Warning: Could not download Lizard: {e}")
    elif args.dataset == "pannuke":
        download_pannuke(args.output_dir, args.folds)
    elif args.dataset == "consep":
        download_consep(args.output_dir)
    elif args.dataset == "monusac":
        download_monusac(args.output_dir)
    elif args.dataset == "lizard":
        download_lizard(args.output_dir)

    print("\n" + "="*70)
    print("✅ DOWNLOAD COMPLETE")
    print("="*70)
    print(f"\nDatasets saved to: {args.output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Convert annotations: python scripts/evaluation/convert_annotations.py")
    print("  2. Run evaluation: python scripts/evaluation/evaluate_ground_truth.py")


if __name__ == "__main__":
    main()
