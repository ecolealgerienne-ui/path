# Plan d'Intégration IHM - Normalisation HV Maps

**Date**: 2025-12-21
**Contexte**: Changement normalisation HV maps [-127, 127] (int8) → [-1, 1] (float32)
**Statut**: ⏳ EN ATTENTE validation training Glandular

---

## 🎯 Objectif

Mettre à jour l'IHM et le pipeline d'inférence pour supporter les modèles entraînés avec HV maps normalisées [-1, 1].

---

## ⚠️ PREREQUIS

**NE PAS IMPLÉMENTER** avant d'avoir validé que l'entraînement avec NEW data fonctionne:

- [ ] Entraînement Glandular terminé (50 epochs)
- [ ] Métriques validation OK (HV MSE < 0.015)
- [ ] Test inférence sur quelques images réussit
- [ ] Checkpoint `hovernet_glandular_FIXED_best.pth` disponible

**Une fois validé** → Procéder avec ce plan.

---

## 🔍 Fichiers Impactés

### 1. Inférence Core

| Fichier | Impact | Action |
|---------|--------|--------|
| `src/inference/hoptimus_hovernet.py` | HV predictions range | Vérifier range [-1, 1] |
| `src/inference/optimus_gate_inference.py` | HV post-processing | Ajuster seuils watershed |
| `src/inference/optimus_gate_inference_multifamily.py` | Multi-family HV | Vérifier cohérence |
| `src/models/hovernet_decoder.py` | Output activation | Vérifier tanh() final |

### 2. Post-Processing

| Fichier | Impact | Action |
|---------|--------|--------|
| `src/inference/hoptimus_hovernet.py` | `watershed_instance_separation()` | Ajuster gradient thresholds |
| (Potentiel) utility watershed | Seuils Sobel | Vérifier edge_threshold |

### 3. Visualisation

| Fichier | Impact | Action |
|---------|--------|--------|
| `scripts/demo/gradio_demo.py` | HV heatmaps | Mettre à jour vmin/vmax |
| (Fonction visualize) | Colormaps | Vérifier échelle [-1, 1] |

### 4. Métriques Cliniques

| Fichier | Impact | Action |
|---------|--------|--------|
| `src/metrics/morphometry.py` | Gradients HV | Vérifier si utilisé |

---

## 📝 Checklist d'Implémentation

### Phase 1: Vérification Inférence (30 min)

```bash
# 1. Charger nouveau checkpoint
checkpoint = torch.load("models/checkpoints_FIXED/hovernet_glandular_best.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# 2. Test sur 1 image
hv_pred, np_pred, nt_pred = model(features)

# 3. Vérifier range
print(f"HV range: [{hv_pred.min():.3f}, {hv_pred.max():.3f}]")
# Attendu: [-1.000, 1.000] ou proche

# 4. Vérifier activation finale du décodeur
# Dans hovernet_decoder.py, ligne ~200:
# self.hv_head = nn.Sequential(
#     nn.Conv2d(...),
#     nn.Tanh()  # ← DOIT être présent pour [-1, 1]
# )
```

**Critère validation**:
- ✅ HV range dans [-1.1, 1.1] (tolérance float)
- ✅ `nn.Tanh()` présent dans HV head
- ✅ Pas de `* 127` ou `/127` dans l'inférence

### Phase 2: Ajustement Post-Processing (1h)

#### 2.1. Fonction Watershed

**Fichier**: `src/inference/hoptimus_hovernet.py` (ou équivalent)

```python
def watershed_instance_separation(hv_map: np.ndarray, np_mask: np.ndarray):
    """
    Sépare les instances via watershed sur gradients HV.

    Args:
        hv_map: (2, H, W) HV predictions normalisées [-1, 1]
        np_mask: (H, W) masque binaire noyaux
    """
    # Calculer gradient
    sobel_h = cv2.Sobel(hv_map[0], cv2.CV_64F, 1, 0, ksize=5)
    sobel_v = cv2.Sobel(hv_map[1], cv2.CV_64F, 0, 1, ksize=5)
    gradient = np.sqrt(sobel_h**2 + sobel_v**2)

    # AVANT (OLD): edge_threshold = 0.5 (échelle [-127, 127])
    # APRÈS (NEW): edge_threshold = 0.05 (échelle [-1, 1])

    # Normaliser gradient
    if gradient.max() > 0:
        gradient = gradient / gradient.max()

    # Seuil pour détecter frontières
    edge_threshold = 0.1  # ← À AJUSTER si nécessaire
    edges = gradient > edge_threshold

    # Watershed standard
    # ...
```

**Action**:
1. Lire fonction actuelle `watershed_instance_separation()`
2. Identifier seuils hardcodés (edge_threshold, dist_threshold)
3. Tester sur 5-10 images avec NEW model
4. Ajuster si nécessaire

**Méthode d'ajustement**:
```python
# Script de tuning
for edge_thresh in [0.05, 0.1, 0.15, 0.2]:
    instances = watershed_instance_separation(hv_pred, np_mask, edge_thresh)
    score = compare_to_ground_truth(instances, gt_instances)
    print(f"Threshold {edge_thresh}: F1={score}")
```

#### 2.2. Vérifier Gradio Demo

**Fichier**: `scripts/demo/gradio_demo.py`

Chercher visualisations HV:
```python
# AVANT (si échelle incorrecte):
plt.imshow(hv_map[0], cmap='RdBu_r', vmin=-127, vmax=127)

# APRÈS (correct):
plt.imshow(hv_map[0], cmap='RdBu_r', vmin=-1, vmax=1)
```

**Action**:
```bash
# 1. Grep toutes les visualisations HV
grep -n "imshow.*hv" scripts/demo/gradio_demo.py

# 2. Vérifier vmin/vmax
# 3. Mettre à jour si nécessaire
```

### Phase 3: Métriques Morphométriques (30 min)

**Fichier**: `src/metrics/morphometry.py`

```bash
# Vérifier si gradients HV utilisés
grep -n "gradient\|sobel\|hv" src/metrics/morphometry.py
```

**Si gradients HV utilisés**:
- Vérifier échelle attendue
- Ajuster seuils si nécessaire
- Tester sur images de référence

**Si gradients HV NON utilisés**:
- ✅ Aucune action requise

### Phase 4: Tests de Non-Régression (1h)

#### 4.1. Test Inférence End-to-End

```bash
# Script de test
cat > scripts/validation/test_inference_fixed.py << 'EOF'
#!/usr/bin/env python3
"""
Test que l'inférence avec NEW model fonctionne correctement.
"""

import torch
import numpy as np
from pathlib import Path
from src.inference.optimus_gate_inference_multifamily import OptimusGateInferenceMultiFamily

def test_inference():
    """Test inférence avec checkpoint FIXED."""

    # Charger modèle FIXED
    model = OptimusGateInferenceMultiFamily(
        checkpoint_dir="models/checkpoints_FIXED",
        device="cuda"
    )

    # Charger image test
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Inférence
    result = model.predict(image)

    # Vérifications
    assert 'hv_pred' in result, "HV predictions manquantes"

    hv_pred = result['hv_pred']
    print(f"HV range: [{hv_pred.min():.3f}, {hv_pred.max():.3f}]")

    # Vérifier range [-1, 1]
    assert hv_pred.min() >= -1.1, f"HV min trop bas: {hv_pred.min()}"
    assert hv_pred.max() <= 1.1, f"HV max trop haut: {hv_pred.max()}"

    print("✅ Test inférence PASSED")
    print(f"   HV range OK: [{hv_pred.min():.3f}, {hv_pred.max():.3f}]")
    print(f"   Instances détectées: {result.get('n_cells', 'N/A')}")

if __name__ == "__main__":
    test_inference()
EOF

chmod +x scripts/validation/test_inference_fixed.py
python scripts/validation/test_inference_fixed.py
```

#### 4.2. Test Comparatif OLD vs NEW

```python
# Comparer sur 10 images de test
# Vérifier que NEW détecte plus d'instances (attendu: ~5x plus)
```

### Phase 5: Documentation (30 min)

#### 5.1. Mettre à jour CLAUDE.md

```markdown
## Normalisation HV Maps (MISE À JOUR 2025-12-21)

### ⚠️ CHANGEMENT MAJEUR

**AVANT (versions ≤ 2025-12-20)**:
- HV maps stockées en int8 [-127, 127]
- ❌ NON conforme HoVer-Net

**APRÈS (versions ≥ 2025-12-21)**:
- HV maps normalisées float32 [-1, 1]
- ✅ Conforme HoVer-Net (Graham et al., 2019)

### Impact sur l'Inférence

Modèles entraînés avec NEW data:
```python
hv_pred: Tensor[B, 2, H, W]  # Range: [-1, 1]
```

Post-processing watershed:
- Seuils ajustés pour échelle [-1, 1]
- Amélioration séparation instances (ratio 1.63x)

### Rétro-Compatibilité

❌ **Modèles OLD incompatibles avec NEW data**
❌ **Modèles NEW incompatibles avec OLD data**

→ Ré-entraînement OBLIGATOIRE pour tous les modèles.
```

#### 5.2. Mettre à jour README Demo

```markdown
## HV Maps Visualization

Les cartes HV (Horizontal/Vertical) affichent les gradients de distance
au centre des noyaux, normalisés à [-1, 1].

**Interprétation**:
- Rouge (-1): Pixel à gauche/haut du centre
- Bleu (+1): Pixel à droite/bas du centre
- Gradient fort (jaune): Frontière entre cellules
```

---

## 🧪 Critères de Validation Finale

Avant de merger en production:

- [ ] ✅ Inférence fonctionne (HV range [-1, 1])
- [ ] ✅ Watershed détecte instances correctement
- [ ] ✅ Gradio demo affiche HV maps correctement
- [ ] ✅ Tests non-régression passent
- [ ] ✅ Documentation à jour
- [ ] ✅ Checkpoint FIXED déployé

---

## 📊 Timeline Estimée

| Phase | Durée | Dépendance |
|-------|-------|------------|
| **PRÉREQUIS** | - | Training Glandular validé |
| Phase 1: Vérification | 30 min | Checkpoint disponible |
| Phase 2: Post-processing | 1h | Phase 1 OK |
| Phase 3: Morphométrie | 30 min | Phase 2 OK |
| Phase 4: Tests | 1h | Phase 3 OK |
| Phase 5: Documentation | 30 min | Phase 4 OK |
| **TOTAL** | **~3.5h** | Après validation training |

---

## 🚨 Points de Vigilance

### 1. Gradients Sobel
```python
# Sobel calcule dérivées → sensible à l'échelle
# Vérifier que seuils edge_threshold sont adaptés
```

### 2. Watershed Seeds
```python
# Seeds basés sur local_maxima(gradient)
# Vérifier que threshold détection seeds est adapté
```

### 3. Visualisation Colormaps
```python
# vmin/vmax doivent correspondre à [-1, 1]
# Sinon, visualisation saturée ou trop pale
```

### 4. Backward Compatibility
```python
# Si ancien checkpoint chargé par erreur:
# Ajouter vérification version dans checkpoint
checkpoint = {
    'model_state_dict': ...,
    'hv_normalization': 'float32_normalized',  # Nouveau champ
    'version': '2025-12-21'
}
```

---

## 📝 Checklist Finale (À compléter après implémentation)

- [ ] Code inférence vérifié
- [ ] Watershed ajusté et testé
- [ ] Visualisations corrigées
- [ ] Métriques morphométriques OK
- [ ] Tests non-régression passent
- [ ] CLAUDE.md mis à jour
- [ ] README demo mis à jour
- [ ] Commit avec message clair
- [ ] PR créée (si applicable)
- [ ] Validation par pathologiste (si applicable)

---

**Créé le**: 2025-12-21
**Par**: Claude (Suite à découverte bug normalisation HV)
**Statut**: ⏳ DRAFT - En attente validation training
