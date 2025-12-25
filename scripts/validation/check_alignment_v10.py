#!/usr/bin/env python3
"""
Diagnostic: Vérifie l'alignement image/mask dans epidermal v10.

Affiche les 5 premiers échantillons pour vérification visuelle.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load v10 data
data = np.load('data/family_FIXED/epidermal_data_FIXED_v9_NUCLEI_ONLY.npz')

images = data['images']
np_targets = data['np_targets']
nt_targets = data['nt_targets']

print("=" * 80)
print("🔍 DIAGNOSTIC ALIGNEMENT IMAGE/MASK v10")
print("=" * 80)

# Vérifier 5 premiers échantillons
n_check = min(5, len(images))

fig, axes = plt.subplots(n_check, 4, figsize=(16, 4 * n_check))

for i in range(n_check):
    img = images[i]
    np_target = np_targets[i]
    nt_target = nt_targets[i]

    # Image originale
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Sample {i}: Image H&E")
    axes[i, 0].axis('off')

    # NP target (noyaux détectés)
    axes[i, 1].imshow(np_target, cmap='gray')
    axes[i, 1].set_title(f"NP Target (coverage: {np_target.mean()*100:.1f}%)")
    axes[i, 1].axis('off')

    # NT target (types de noyaux)
    axes[i, 2].imshow(nt_target, cmap='tab10', vmin=0, vmax=5)
    axes[i, 2].set_title(f"NT Target (max: {nt_target.max()})")
    axes[i, 2].axis('off')

    # Overlay (image + NP en rouge)
    overlay = img.copy()
    overlay[np_target > 0, 0] = 255  # Rouge pour noyaux
    axes[i, 3].imshow(overlay)
    axes[i, 3].set_title("Overlay (noyaux en rouge)")
    axes[i, 3].axis('off')

    # Stats
    nt_unique, nt_counts = np.unique(nt_target, return_counts=True)
    nt_dist = {int(cls): int(cnt) for cls, cnt in zip(nt_unique, nt_counts)}

    print(f"\nSample {i}:")
    print(f"  NP coverage: {np_target.mean()*100:.2f}%")
    print(f"  NT distribution: {nt_dist}")

    # VÉRIFICATION CRITIQUE: Les noyaux rouges doivent être sur les régions sombres
    # de l'image H&E (noyaux = violet/bleu foncé)
    if np_target.mean() < 0.05:
        print(f"  ⚠️  WARNING: NP coverage très faible (<5%)")

    # NT doit être majoritairement classe 4 (Dead/Epithelial proxy) pour epidermal
    if nt_target.max() < 4:
        print(f"  ⚠️  WARNING: NT max < 4 (pas de classe Epithelial/Dead)")

plt.tight_layout()
output_file = Path("results/alignment_check_v10.png")
output_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n💾 Sauvegardé: {output_file}")

# Distribution globale NT
print("\n" + "=" * 80)
print("📊 DISTRIBUTION NT GLOBALE:")
print("=" * 80)

nt_unique, nt_counts = np.unique(nt_targets, return_counts=True)
total = nt_targets.size

for cls, cnt in zip(nt_unique, nt_counts):
    pct = cnt / total * 100
    class_names = {0: "Background", 1: "Neoplastic", 2: "Inflammatory",
                   3: "Connective", 4: "Dead/Epithelial", 5: "ERROR"}
    name = class_names.get(int(cls), f"Unknown({cls})")
    print(f"  Classe {cls} ({name}): {cnt:>12} pixels ({pct:>6.2f}%)")

# VÉRIFICATION CRITIQUE
np_coverage = np_targets.mean() * 100
nt_nuclei = nt_counts[nt_unique > 0].sum() if len(nt_unique) > 1 else 0
nt_nuclei_pct = nt_nuclei / total * 100

print(f"\n⚠️  VÉRIFICATION COHÉRENCE:")
print(f"  NP coverage:     {np_coverage:.2f}%")
print(f"  NT nuclei (1-4): {nt_nuclei_pct:.2f}%")
print(f"  Différence:      {abs(np_coverage - nt_nuclei_pct):.2f}%")

if abs(np_coverage - nt_nuclei_pct) > 2:
    print(f"\n❌ MISMATCH CRITIQUE!")
    print(f"   NP détecte {np_coverage:.1f}% de noyaux")
    print(f"   NT classe {nt_nuclei_pct:.1f}% comme noyaux")
    print(f"   → Incohérence de {abs(np_coverage - nt_nuclei_pct):.1f}%")
else:
    print(f"\n✅ Cohérence NP/NT OK")

print("\n" + "=" * 80)
print("🎯 CONSIGNES VÉRIFICATION VISUELLE:")
print("=" * 80)
print("  1. Les noyaux rouges (overlay) doivent correspondre aux zones violet/bleu")
print("     foncé de l'image H&E (= noyaux)")
print("  2. Si les noyaux rouges sont sur du fond blanc/rose (cytoplasme/stroma),")
print("     c'est un DÉSALIGNEMENT image/mask!")
print("  3. NT doit avoir ~15% classe 4 (Dead/Epithelial), pas 0.1%")
print("\n  Ouvrir: results/alignment_check_v10.png")
