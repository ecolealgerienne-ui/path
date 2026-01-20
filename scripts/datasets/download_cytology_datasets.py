#!/usr/bin/env python3
"""
Download Open Source Cytology Datasets

Datasets téléchargés:
- Herlev (Cervix Pap smear): 917 images
- TB-PANDA (Thyroid FNA): ~10,000 images
- SIPaKMeD (Cervix): 4,049 images
- ISBI 2014 (Breast mitoses): 1,200 images

Usage:
    python scripts/datasets/download_cytology_datasets.py --all
    python scripts/datasets/download_cytology_datasets.py --dataset herlev
"""

import os
import sys
import argparse
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import subprocess

# Directories
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, destination, desc="Downloading"):
    """
    Download file with progress bar

    Args:
        url: Download URL
        destination: Local file path
        desc: Progress bar description
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

def extract_archive(archive_path, extract_to):
    """Extract ZIP or TAR archive"""
    print(f"📦 Extracting {archive_path.name}...")

    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)

    print(f"✅ Extracted to {extract_to}")

def download_herlev():
    """
    Herlev Dataset - Cervical Pap Smear (917 images)

    Reference:
    - Paper: Jantzen et al. (2005)
    - URL: http://mde-lab.aegean.gr/index.php/downloads
    - Classes: Normal, Light Dysplasia, Moderate Dysplasia, Severe Dysplasia, Carcinoma
    """
    print("\n" + "="*60)
    print("📥 HERLEV DATASET (Cervical Pap Smear)")
    print("="*60)

    dest_dir = DATA_DIR / "herlev"
    dest_dir.mkdir(exist_ok=True)

    # Note: Herlev dataset requires manual download from website
    # URL: http://mde-lab.aegean.gr/downloads/Herlev_dataset.zip

    print("⚠️  MANUAL DOWNLOAD REQUIRED")
    print("\nSteps:")
    print("1. Visit: http://mde-lab.aegean.gr/index.php/downloads")
    print("2. Download: Herlev_dataset.zip")
    print(f"3. Extract to: {dest_dir}")
    print("\n💡 Alternative: Use `wget` if direct link available")

    # If user has already downloaded manually
    if (dest_dir / "images").exists():
        print(f"\n✅ Herlev dataset already exists at {dest_dir}")
        return True

    return False

def download_tb_panda():
    """
    TB-PANDA Dataset - Thyroid Cytopathology (~10,000 images)

    Reference:
    - Paper: Sanyal et al. (2018)
    - GitHub: https://github.com/ncbi/TB-PANDA
    - Classes: Bethesda I-VI
    """
    print("\n" + "="*60)
    print("📥 TB-PANDA DATASET (Thyroid FNA)")
    print("="*60)

    dest_dir = DATA_DIR / "tb_panda"

    if dest_dir.exists():
        print(f"✅ TB-PANDA already exists at {dest_dir}")
        return True

    print("📦 Cloning TB-PANDA repository...")

    try:
        subprocess.run([
            "git", "clone",
            "https://github.com/ncbi/TB-PANDA.git",
            str(dest_dir)
        ], check=True)

        print(f"✅ TB-PANDA downloaded to {dest_dir}")
        return True

    except subprocess.CalledProcessError:
        print("❌ Failed to clone TB-PANDA")
        print("💡 Try manually: git clone https://github.com/ncbi/TB-PANDA.git data/raw/tb_panda")
        return False

def download_sipakmed():
    """
    SIPaKMeD Dataset - Cervical Pap Smear (4,049 images)

    Reference:
    - Paper: Plissiti et al. (2018)
    - URL: https://www.cs.uoi.gr/~marina/sipakmed.html
    - Classes: 5 cell types
    """
    print("\n" + "="*60)
    print("📥 SIPaKMeD DATASET (Cervical Pap Smear)")
    print("="*60)

    dest_dir = DATA_DIR / "sipakmed"
    dest_dir.mkdir(exist_ok=True)

    # Note: SIPaKMeD requires registration/request
    print("⚠️  REGISTRATION REQUIRED")
    print("\nSteps:")
    print("1. Visit: https://www.cs.uoi.gr/~marina/sipakmed.html")
    print("2. Fill registration form")
    print("3. Download dataset after approval")
    print(f"4. Extract to: {dest_dir}")

    if (dest_dir / "images").exists():
        print(f"\n✅ SIPaKMeD dataset already exists at {dest_dir}")
        return True

    return False

def download_isbi_2014():
    """
    ISBI 2014 Challenge - Mitosis Detection in Breast Histology (~1,200 images)

    Reference:
    - Challenge: https://mitos-atypia-14.grand-challenge.org/
    - Note: Plus histologie que cytologie, mais utile pour mitoses
    """
    print("\n" + "="*60)
    print("📥 ISBI 2014 DATASET (Breast Mitoses)")
    print("="*60)

    dest_dir = DATA_DIR / "isbi_2014_mitoses"
    dest_dir.mkdir(exist_ok=True)

    print("⚠️  REGISTRATION REQUIRED")
    print("\nSteps:")
    print("1. Visit: https://mitos-atypia-14.grand-challenge.org/")
    print("2. Create account")
    print("3. Download training/test sets")
    print(f"4. Extract to: {dest_dir}")

    if (dest_dir / "images").exists():
        print(f"\n✅ ISBI 2014 dataset already exists at {dest_dir}")
        return True

    return False

def search_kaggle_datasets():
    """
    Search Kaggle for cytology datasets

    Requires: pip install kaggle
    Requires: Kaggle API token (~/.kaggle/kaggle.json)
    """
    print("\n" + "="*60)
    print("🔍 SEARCHING KAGGLE DATASETS")
    print("="*60)

    try:
        import kaggle
    except ImportError:
        print("⚠️  Kaggle library not installed")
        print("💡 Install: pip install kaggle")
        print("💡 Setup: https://www.kaggle.com/docs/api")
        return

    keywords = [
        "cervical cytology",
        "pap smear",
        "thyroid cytology",
        "thyroid FNA",
        "urine cytology",
        "bladder cancer cytology",
        "bethesda system",
        "paris system urology",
    ]

    print(f"🔎 Searching {len(keywords)} keywords...\n")

    for keyword in keywords:
        print(f"📌 '{keyword}':")
        try:
            datasets = kaggle.api.dataset_list(search=keyword, page_size=5)

            if not datasets:
                print("   No datasets found")
            else:
                for ds in datasets[:5]:
                    print(f"   • {ds.ref}")
                    print(f"     Title: {ds.title}")
                    print(f"     Size: {ds.size}")
                    print(f"     Download: kaggle datasets download -d {ds.ref}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        print()

def download_mendeley_cervical():
    """
    Mendeley Cervical Cancer Dataset

    Reference:
    - URL: https://data.mendeley.com/datasets
    - Search: "cervical cancer cytology"
    """
    print("\n" + "="*60)
    print("📥 MENDELEY CERVICAL CANCER DATASETS")
    print("="*60)

    print("🔗 Visit: https://data.mendeley.com/datasets")
    print("🔎 Search keywords:")
    print("   - cervical cancer cytology")
    print("   - pap smear classification")
    print("   - liquid-based cytology")

    print("\n💡 Datasets may require registration and citation")

def print_summary():
    """Print summary of all datasets"""
    print("\n" + "="*60)
    print("📊 CYTOLOGY DATASETS SUMMARY")
    print("="*60)

    datasets = [
        {
            "name": "Herlev",
            "organ": "Cervix",
            "samples": "917",
            "type": "Pap Smear",
            "classes": "5 (Normal → Carcinoma)",
            "status": "Manual download",
            "url": "http://mde-lab.aegean.gr"
        },
        {
            "name": "TB-PANDA",
            "organ": "Thyroid",
            "samples": "~10,000",
            "type": "FNA",
            "classes": "Bethesda I-VI",
            "status": "Git clone",
            "url": "https://github.com/ncbi/TB-PANDA"
        },
        {
            "name": "SIPaKMeD",
            "organ": "Cervix",
            "samples": "4,049",
            "type": "Pap Smear",
            "classes": "5 cell types",
            "status": "Registration required",
            "url": "https://www.cs.uoi.gr/~marina/sipakmed.html"
        },
        {
            "name": "ISBI 2014",
            "organ": "Breast",
            "samples": "~1,200",
            "type": "Mitoses",
            "classes": "Binary (Mitosis/No)",
            "status": "Registration required",
            "url": "https://mitos-atypia-14.grand-challenge.org/"
        },
        {
            "name": "Kaggle Datasets",
            "organ": "Multi",
            "samples": "Variable",
            "type": "Various",
            "classes": "Variable",
            "status": "Search required",
            "url": "https://www.kaggle.com/datasets"
        }
    ]

    print("\n| Dataset | Organ | Samples | Type | Classes | Status |")
    print("|---------|-------|---------|------|---------|--------|")
    for ds in datasets:
        print(f"| {ds['name']} | {ds['organ']} | {ds['samples']} | {ds['type']} | {ds['classes']} | {ds['status']} |")

    print("\n📁 Download directory: data/raw/")
    print("📖 Documentation: docs/V14_CYTOLOGY_STANDALONE_STRATEGY.md")

def main():
    parser = argparse.ArgumentParser(
        description="Download open source cytology datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=["herlev", "tb_panda", "sipakmed", "isbi_2014", "kaggle", "all"],
        help="Dataset to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available datasets"
    )
    parser.add_argument(
        "--search-kaggle",
        action="store_true",
        help="Search Kaggle for cytology datasets"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of all datasets"
    )

    args = parser.parse_args()

    print("="*60)
    print("🧬 CYTOLOGY DATASETS DOWNLOADER")
    print("="*60)

    if args.summary or (not args.dataset and not args.all and not args.search_kaggle):
        print_summary()
        return

    if args.search_kaggle:
        search_kaggle_datasets()
        return

    datasets_to_download = []

    if args.all:
        datasets_to_download = ["herlev", "tb_panda", "sipakmed", "isbi_2014"]
    elif args.dataset:
        datasets_to_download = [args.dataset]

    success_count = 0
    total_count = len(datasets_to_download)

    for dataset in datasets_to_download:
        if dataset == "herlev":
            result = download_herlev()
        elif dataset == "tb_panda":
            result = download_tb_panda()
        elif dataset == "sipakmed":
            result = download_sipakmed()
        elif dataset == "isbi_2014":
            result = download_isbi_2014()
        elif dataset == "kaggle":
            search_kaggle_datasets()
            result = True

        if result:
            success_count += 1

    print("\n" + "="*60)
    print(f"📊 DOWNLOAD SUMMARY: {success_count}/{total_count} completed")
    print("="*60)

    print("\n💡 Next steps:")
    print("1. Check data/raw/ for downloaded datasets")
    print("2. Run preprocessing: python scripts/datasets/preprocess_cytology.py")
    print("3. Verify datasets: python scripts/datasets/verify_datasets.py")

if __name__ == "__main__":
    main()
