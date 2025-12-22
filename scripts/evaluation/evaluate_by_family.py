#!/usr/bin/env python3
"""
Évalue chaque famille HoVer-Net sur ses images appropriées.

Au lieu d'utiliser OrganHead pour router, ce script teste chaque famille
sur un dataset pré-organisé par type de tissu.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

FAMILIES = ["glandular", "digestive", "urologic", "respiratory", "epidermal"]


def run_evaluation(
    dataset_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    family: str,
    num_samples: int = None,
):
    """Run evaluation for a single family."""
    family_dir = dataset_dir / family
    family_output = output_dir / family

    if not family_dir.exists():
        print(f"⚠️ Skipping {family}: directory not found ({family_dir})")
        return None

    # Count images
    npz_files = list(family_dir.glob("*.npz"))
    n_images = len(npz_files)

    if n_images == 0:
        print(f"⚠️ Skipping {family}: no images found")
        return None

    # Limit samples if requested
    if num_samples:
        n_images = min(n_images, num_samples)

    print(f"\n{'='*70}")
    print(f"EVALUATING {family.upper()}")
    print(f"{'='*70}")
    print(f"Images: {n_images}")
    print(f"Output: {family_output}")

    # Build command
    cmd = [
        sys.executable,
        "scripts/evaluation/evaluate_ground_truth.py",
        "--dataset_dir", str(family_dir),
        "--checkpoint_dir", str(checkpoint_dir),
        "--output_dir", str(family_output),
        "--force_family", family,
    ]

    if num_samples:
        cmd.extend(["--num_samples", str(num_samples)])

    # Run evaluation
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        # Parse metrics from output
        output = result.stdout
        metrics = {}

        for line in output.split('\n'):
            if 'Dice:' in line and 'target' in line:
                parts = line.split()
                metrics['dice'] = float(parts[1])
            elif 'AJI:' in line and 'target' in line:
                parts = line.split()
                metrics['aji'] = float(parts[1])
            elif 'PQ:' in line and 'target' in line:
                parts = line.split()
                metrics['pq'] = float(parts[1])
            elif 'Rappel:' in line:
                parts = line.split()
                metrics['recall'] = float(parts[1].rstrip('%'))

        metrics['n_images'] = n_images
        metrics['family'] = family

        return metrics

    except subprocess.CalledProcessError as e:
        print(f"❌ Error evaluating {family}:")
        print(e.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all families on their specific test images"
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Directory with family subdirectories (glandular/, digestive/, etc.)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Directory with family checkpoints"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit number of samples per family (for quick testing)"
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=FAMILIES,
        default=FAMILIES,
        help="Families to evaluate (default: all)"
    )

    args = parser.parse_args()

    print("="*70)
    print("FAMILY-SPECIFIC EVALUATION")
    print("="*70)
    print(f"Dataset:    {args.dataset_dir}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Output:     {args.output_dir}")
    print(f"Families:   {', '.join(args.families)}")

    # Run evaluations
    results = {}
    for family in args.families:
        metrics = run_evaluation(
            args.dataset_dir,
            args.checkpoint_dir,
            args.output_dir,
            family,
            args.num_samples
        )
        if metrics:
            results[family] = metrics

    # Summary report
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)

    if not results:
        print("❌ No families evaluated successfully")
        return

    # Table header
    print(f"\n{'Family':<12} {'Images':<8} {'Dice':<8} {'AJI':<8} {'PQ':<8} {'Recall':<8}")
    print("-"*70)

    total_images = 0
    avg_dice = []
    avg_aji = []
    avg_pq = []
    avg_recall = []

    for family in FAMILIES:
        if family not in results:
            continue

        m = results[family]
        total_images += m['n_images']

        dice = m.get('dice', 0)
        aji = m.get('aji', 0)
        pq = m.get('pq', 0)
        recall = m.get('recall', 0)

        avg_dice.append(dice)
        avg_aji.append(aji)
        avg_pq.append(pq)
        avg_recall.append(recall)

        print(f"{family.capitalize():<12} {m['n_images']:<8} "
              f"{dice:<8.4f} {aji:<8.4f} {pq:<8.4f} {recall:<8.2f}%")

    # Averages
    print("-"*70)
    print(f"{'AVERAGE':<12} {total_images:<8} "
          f"{sum(avg_dice)/len(avg_dice):<8.4f} "
          f"{sum(avg_aji)/len(avg_aji):<8.4f} "
          f"{sum(avg_pq)/len(avg_pq):<8.4f} "
          f"{sum(avg_recall)/len(avg_recall):<8.2f}%")

    # Targets
    print("\n" + "="*70)
    print("TARGETS")
    print("="*70)
    print("Dice:   > 0.95")
    print("AJI:    > 0.80")
    print("PQ:     > 0.70")
    print("Recall: > 90%")

    # Save JSON summary
    summary_file = args.output_dir / "summary_by_family.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Summary saved: {summary_file}")


if __name__ == "__main__":
    main()
