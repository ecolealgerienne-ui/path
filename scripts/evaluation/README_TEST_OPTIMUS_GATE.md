# 🧪 Scripts de Test OptimusGate Multi-Famille

Suite de scripts pour valider le pipeline OptimusGate complet avec les 5 familles HoVer-Net.

---

## 📋 Scripts Disponibles

### 1. **validate_all_checkpoints.py** — Validation rapide

**Objectif:** Vérifier que tous les checkpoints se chargent correctement et extraire les métriques d'entraînement.

**Usage:**
```bash
python scripts/evaluation/validate_all_checkpoints.py \
    --checkpoints_dir models/checkpoints
```

**Sortie:**
```
✅ 5/5 checkpoints valides

📊 Tableau récapitulatif:
Famille         Epoch    NP Dice    HV MSE     NT Acc
---------------------------------------------------------------
Glandular       50       0.9536     0.0426     0.9002
Digestive       50       0.9610     0.0533     0.8802
Urologic        50       0.9304     0.0485     0.9098
Epidermal       50       0.9519     0.2965     0.8960
Respiratory     50       0.9384     0.2519     0.9032
```

**Temps:** ~5 secondes

---

### 2. **test_visual_samples.py** — Test visuel

**Objectif:** Générer des visualisations comparant prédictions vs ground truth pour chaque famille.

**Usage:**
```bash
python scripts/evaluation/test_visual_samples.py \
    --data_dir /home/amar/data/PanNuke \
    --checkpoints_dir models/checkpoints \
    --output_dir results/visual_test \
    --fold 2 \
    --n_per_family 3 \
    --device cuda
```

**Paramètres:**
- `--fold`: Fold PanNuke à utiliser (0, 1, 2) — défaut: 2 (validation)
- `--n_per_family`: Nombre d'échantillons par famille — défaut: 3
- `--device`: Device PyTorch (cuda/cpu) — défaut: cuda

**Sortie:**
- Images PNG comparatives (image H&E + GT + prédiction)
- Nommage: `{famille}_{idx}_{organe}.png`
- Exemple: `glandular_1_Breast.png`, `digestive_2_Colon.png`

**Temps:** ~1-2 min pour 15 images (3 par famille)

---

### 3. **test_optimus_gate_multifamily.py** — Test complet

**Objectif:** Tester l'ensemble du pipeline OptimusGate avec routage OrganHead → Famille et métriques complètes.

**Usage:**
```bash
python scripts/evaluation/test_optimus_gate_multifamily.py \
    --data_dir /home/amar/data/PanNuke \
    --checkpoints_dir models/checkpoints \
    --fold 2 \
    --n_samples 100 \
    --output_dir results/optimus_gate_test \
    --device cuda
```

**Paramètres:**
- `--fold`: Fold PanNuke à tester — défaut: 2
- `--n_samples`: Nombre d'échantillons à tester — défaut: 50
- `--device`: Device PyTorch — défaut: cuda

**Sortie:**
- Fichier JSON `test_results_YYYYMMDD_HHMMSS.json` contenant:
  - Précision de routage OrganHead → Famille
  - Métriques NP/HV/NT par famille
  - Résultats détaillés par échantillon

**Exemple de sortie:**
```json
{
  "metadata": {
    "fold": 2,
    "n_samples": 100,
    "timestamp": "2025-12-22T18:30:00"
  },
  "routing": {
    "organ_accuracy": 0.99,
    "family_accuracy": 1.0
  },
  "metrics_by_family": {
    "glandular": {
      "n_samples": 35,
      "dice_mean": 0.9540,
      "dice_std": 0.0184,
      "hv_mse_mean": 0.0430,
      "hv_mse_std": 0.0104,
      "nt_acc_mean": 0.9010,
      "nt_acc_std": 0.0229
    },
    ...
  }
}
```

**Temps:** ~5-10 min pour 100 échantillons (GPU)

---

## 🚀 Workflow Recommandé

### **Étape 1: Validation Rapide**

Vérifier que tous les checkpoints sont valides:

```bash
python scripts/evaluation/validate_all_checkpoints.py
```

➡️ Si ✅, passer à l'étape 2. Sinon, vérifier les chemins.

---

### **Étape 2: Test Visuel**

Générer quelques visualisations pour inspection manuelle:

```bash
python scripts/evaluation/test_visual_samples.py \
    --data_dir /home/amar/data/PanNuke \
    --fold 2 \
    --n_per_family 5
```

➡️ Ouvrir les images dans `results/visual_test/` et vérifier visuellement que les prédictions sont cohérentes.

---

### **Étape 3: Test Complet**

Évaluer quantitativement sur 100+ échantillons:

```bash
python scripts/evaluation/test_optimus_gate_multifamily.py \
    --data_dir /home/amar/data/PanNuke \
    --fold 2 \
    --n_samples 100
```

➡️ Analyser le fichier JSON de sortie pour:
- Vérifier que `family_accuracy > 0.99` (routage correct)
- Comparer les métriques par famille avec les résultats d'entraînement

---

## 📊 Résultats Attendus (Référence)

Basé sur l'entraînement des 5 familles (2025-12-22):

| Famille | NP Dice | HV MSE | NT Acc | Statut |
|---------|---------|--------|--------|--------|
| Glandular | 0.9536 | **0.0426** ✅ | 0.9002 | 🟢 Excellent |
| Digestive | 0.9610 | **0.0533** ✅ | 0.8802 | 🟢 Excellent |
| Urologic | 0.9304 | **0.0485** ✅ | 0.9098 | 🟢 Excellent |
| Epidermal | 0.9519 | 0.2965 ⚠️ | 0.8960 | 🟡 Acceptable |
| Respiratory | 0.9384 | 0.2519 ⚠️ | 0.9032 | 🟡 Acceptable |

**Notes:**
- HV MSE élevé pour Epidermal/Respiratory est **attendu** (peu de samples: 571 et 408)
- NP Dice et NT Acc restent **excellents** même avec peu de données
- Routage OrganHead → Famille devrait être **>99%** (OrganHead accuracy: 99.94%)

---

## 🔧 Dépannage

### Erreur: `ModuleNotFoundError: No module named 'src'`

**Solution:** Ajouter le répertoire racine au PYTHONPATH:

```bash
export PYTHONPATH=/home/user/path:$PYTHONPATH
python scripts/evaluation/validate_all_checkpoints.py
```

Ou utiliser le wrapper:

```bash
cd /home/user/path
python -m scripts.evaluation.validate_all_checkpoints
```

---

### Erreur: `Checkpoint not found`

**Solution:** Vérifier que les checkpoints existent:

```bash
ls -lh models/checkpoints/hovernet_*
```

Attendu:
```
hovernet_glandular_best.pth
hovernet_digestive_best.pth
hovernet_urologic_best.pth
hovernet_epidermal_best.pth
hovernet_respiratory_best.pth
organ_head_best.pth
```

---

### Erreur: `CUDA out of memory`

**Solution:** Réduire `--n_samples` ou utiliser CPU:

```bash
python scripts/evaluation/test_optimus_gate_multifamily.py \
    --n_samples 20 \
    --device cpu
```

---

## 📝 Interprétation des Résultats

### **HV MSE**

| Valeur | Qualité | Impact Clinique |
|--------|---------|-----------------|
| **<0.05** | ✅ Excellent | Séparation instances fiable |
| **0.05-0.15** | ⚠️ Bon | Séparation correcte dans 90%+ des cas |
| **0.15-0.30** | 🟡 Acceptable | Vérification manuelle recommandée pour clusters denses |
| **>0.30** | ❌ Insuffisant | Modèle prédit des valeurs presque plates |

### **NP Dice**

| Valeur | Qualité |
|--------|---------|
| **>0.95** | ✅ Excellent |
| **0.90-0.95** | ⚠️ Bon |
| **<0.90** | 🟡 À améliorer |

### **NT Accuracy**

| Valeur | Qualité |
|--------|---------|
| **>0.90** | ✅ Excellent |
| **0.85-0.90** | ⚠️ Bon |
| **<0.85** | 🟡 À améliorer |

---

## 🎯 Prochaines Étapes

Une fois les tests validés:

1. **Comparer avec CellViT-256** (baseline)
2. **Tester sur images réelles** (hors PanNuke)
3. **Valider avec pathologiste** (retours qualitatifs)
4. **Intégrer dans l'IHM Gradio** (démo interactive)

---

## 📚 Références

- **Graham et al. (2019):** "Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images"
- **PanNuke Dataset:** https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
- **H-optimus-0:** https://huggingface.co/bioptimus/H-optimus-0
