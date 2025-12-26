# Guide Rapide - Régénération Complète

**Situation:** Toutes les données ont été nettoyées
**Objectif:** Régénérer le pipeline complet depuis PanNuke brut jusqu'aux datasets V13-Hybrid prêts pour training

---

## ⚡ Quick Start (1 heure sans training)

### Étape 1: Vérifier PanNuke (30 secondes)

```bash
# Vérifier que PanNuke existe
ls /home/amar/data/PanNuke/fold0/

# Devrait afficher:
# images.npy  masks.npy  types.npy
```

**❌ Si PanNuke n'existe pas:**
1. Télécharger depuis: https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
2. Extraire dans `/home/amar/data/PanNuke/`
3. Structure attendue:
   ```
   /home/amar/data/PanNuke/
   ├── fold0/
   │   ├── images.npy
   │   ├── masks.npy
   │   └── types.npy
   ├── fold1/
   └── fold2/
   ```

### Étape 2: Lancer le Pipeline Automatique (1 heure)

```bash
# Activer environnement
conda activate cellvit

# Lancer le script de régénération complet
bash scripts/utils/regenerate_full_pipeline.sh
```

**Le script va automatiquement:**
1. ✅ Générer Family FIXED data (5 familles, ~30 min)
2. ✅ Préparer V13-Hybrid datasets avec Clean Split (~10 min)
3. ✅ Vérifier Clean Split integrity (~2 min)
4. ✅ Extraire H-features (~5 min)

**Sortie finale attendue:**
```
✅ PIPELINE REGENERATION COMPLETE

📊 GENERATED FILES:

Family FIXED data (data/family_FIXED/):
  ✅ glandular: 1.5G
  ✅ digestive: 1.0G
  ✅ urologic: 500M
  ✅ epidermal: 250M
  ✅ respiratory: 180M

V13-Hybrid datasets (data/family_data_v13_hybrid/):
  ✅ glandular: 1.5G
  ✅ digestive: 1.0G
  ✅ urologic: 500M
  ✅ epidermal: 250M
  ✅ respiratory: 180M

H-Features (data/cache/family_data/):
  ✅ glandular: 15M
  ✅ digestive: 10M
  ✅ urologic: 5M
  ✅ epidermal: 2.5M
  ✅ respiratory: 1.8M
```

### Étape 3: Tester avec une famille (5 minutes)

```bash
# Test rapide sur epidermal (la plus petite)
python scripts/preprocessing/prepare_v13_hybrid_dataset.py --family epidermal
```

**Sortie attendue:**
```
================================================================================
🔒 CREATING CLEAN SPLIT (GROUPED BY SOURCE ID)
================================================================================

📂 Source Image Split:
   Train images: 411 (80.0%)
   Val images:   103 (20.0%)

🔍 Safety Checks:
   ✅ No overlap: 0 crops in both train and val
   ✅ All crops assigned: 2570/2570

✅ CLEAN SPLIT CREATED AND LOCKED TO DISK
```

### Étape 4: Vérifier Clean Split (30 secondes)

```bash
python scripts/validation/verify_clean_split.py \
    --data_file data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz
```

**Sortie attendue:**
```
✅ ALL CHECKS PASSED - Clean Split is VALID!
🎉 This dataset is safe to use for training and validation.
   No data leakage detected.
```

---

## 🚀 Après la Régénération

### Option A: Entraîner Une Famille (40 minutes)

```bash
python scripts/training/train_hovernet_family_v13_hybrid.py \
    --family epidermal \
    --epochs 30 \
    --batch_size 16
```

### Option B: Entraîner Toutes les Familles (3-4 heures)

```bash
for family in glandular digestive urologic epidermal respiratory; do
    python scripts/training/train_hovernet_family_v13_hybrid.py \
        --family $family --epochs 30 --batch_size 16
done
```

### Évaluation (5 minutes par famille)

```bash
python scripts/evaluation/test_v13_hybrid_aji.py \
    --checkpoint models/checkpoints_v13_hybrid/hovernet_epidermal_v13_hybrid_best.pth \
    --family epidermal \
    --n_samples 50
```

**Métriques cibles avec Clean Split:**
- Dice: ~0.93
- **AJI: ≥0.60** (objectif principal)
- Over-seg: ~0.95×

---

## ⚠️ Troubleshooting

### Erreur: "PanNuke data not found"

**Solution:**
```bash
# Vérifier chemin
ls /home/amar/data/PanNuke/

# Si vide, télécharger PanNuke (voir Étape 1)
```

### Erreur: "Conda environment 'cellvit' not found"

**Solution:**
```bash
# Créer environnement
conda create -n cellvit python=3.10 -y
conda activate cellvit

# Installer dépendances
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm transformers scikit-learn scipy numpy opencv-python scikit-image
```

### Erreur: "Out of memory" pendant génération

**Solution:**
```bash
# Générer famille par famille au lieu du script automatique
python scripts/preprocessing/prepare_family_data_FIXED_v12_COHERENT.py \
    --family epidermal --data_dir /home/amar/data/PanNuke

# Puis continuer manuellement avec les autres familles
```

### Erreur: "Clean Split validation FAILED"

**Cause:** Erreur de logique dans le split

**Solution:**
```bash
# Regénérer la famille concernée
python scripts/preprocessing/prepare_v13_hybrid_dataset.py \
    --family epidermal

# Vérifier à nouveau
python scripts/validation/verify_clean_split.py \
    --data_file data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz
```

---

## 📋 Checklist

**Avant de commencer:**
- [ ] PanNuke data téléchargé et décompressé
- [ ] Environnement `cellvit` activé
- [ ] Au moins 11 GB d'espace disque libre
- [ ] Au moins 16 GB RAM système

**Après régénération:**
- [ ] Script `regenerate_full_pipeline.sh` terminé sans erreur
- [ ] Tous les fichiers `*_data_FIXED.npz` créés (5)
- [ ] Tous les fichiers `*_data_v13_hybrid.npz` créés (5)
- [ ] Tous les fichiers `*_h_features_v13.npz` créés (5)
- [ ] Clean Split validation passée pour toutes les familles
- [ ] Conflit NP/NT = 0% pour toutes les familles

**Prêt pour training:**
- [ ] Au moins 1 checkpoint entraîné
- [ ] Métriques validées (AJI ≥0.60)

---

## 📖 Documentation Complète

Pour plus de détails, voir:
- **Pipeline complet:** `docs/REGENERATION_COMPLETE_PIPELINE.md`
- **Clean Split validation:** `docs/CLEAN_SPLIT_IMPLEMENTATION_VALIDATION.md`
- **Architecture V13-Hybrid:** `docs/V13_HYBRID_SPECIFICATIONS.md` (si existe)

---

**Guide Version:** 1.0
**Date:** 2025-12-26
**Temps total estimé:** ~1h (sans training), ~4-5h (avec training)
