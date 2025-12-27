# 🔮 Stratégie Multi-Patches WSI — Production Future

> **STATUS**: Documenté pour référence future (2025-12-27)
> **PRIORITY**: Post-MVP (après validation crops indépendants)

## Contexte

Ce document archive les propositions pour le **stitching de patches WSI** en production, discutées lors de la session de validation V13 Smart Crops.

**Principe actuel validé**: Chaque crop 224×224 = image indépendante
**Extension future**: Reconstruction WSI complète via fusion patches overlapping

---

## Production Workflow Cible

```
WSI 40,000×40,000 pixels
    ↓
Découpage patches 224×224 (stride 200px = overlap 24px)
    ↓
Inférence parallèle sur chaque patch (indépendant)
    ↓
Stitching via NMS + Moyenne pondérée HV
    ↓
Reconstruction instance map complète
```

---

## Stratégie Stitching (Validée Expert)

### 1. Overlap Stride 200px

**Justification:**
- Overlap 24px > diamètre noyau moyen (~15-20px)
- Garantie mathématique: Aucun noyau "entre deux patches"
- Redondance = sécurité diagnostic médical

**Coût:**
- +12% inférences (acceptable)
- +7 min/lame (33 min vs 26 min)

**Bénéfice:**
- AJI: 0.65 (disjoint) → 0.72+ (overlap)
- 0% noyaux perdus aux jointures

### 2. NMS (Non-Maximum Suppression)

**Critère recommandé: COMBINÉ**

```python
def nms_score(instance, patch_bounds):
    """
    Score combiné pour NMS.

    Args:
        instance: Instance détectée
        patch_bounds: Limites du patch source

    Returns:
        score: 70% confiance + 30% bonus distance bord
    """
    confidence = instance.np_score  # Probabilité sigmoid NP
    dist_to_border = min_distance_to_patch_border(instance.centroid, patch_bounds)
    normalized_dist = dist_to_border / (224 / 2)  # Normaliser [0, 1]

    score = confidence * (1 + 0.3 * normalized_dist)
    return score

def stitch_patches_nms(instances_A, instances_B, iou_threshold=0.5):
    """
    Fusionne instances de 2 patches adjacents.

    Args:
        instances_A, instances_B: Listes d'instances détectées
        iou_threshold: Seuil IoU pour considérer "même instance"

    Returns:
        fused_instances: Liste fusionnée sans doublons
    """
    fused = []

    for inst_A in instances_A:
        matched = False
        for inst_B in instances_B:
            iou = compute_iou(inst_A.mask, inst_B.mask)

            if iou > iou_threshold:
                # Même noyau détecté dans les 2 patches
                score_A = nms_score(inst_A, patch_A_bounds)
                score_B = nms_score(inst_B, patch_B_bounds)

                keep = inst_A if score_A > score_B else inst_B
                fused.append(keep)
                matched = True
                break

        if not matched:
            fused.append(inst_A)  # Instance unique à patch A

    # Ajouter instances uniques à patch B
    for inst_B in instances_B:
        if not any(compute_iou(inst_B.mask, f.mask) > iou_threshold for f in fused):
            fused.append(inst_B)

    return fused
```

**Calibration seuil IoU:**
- Commencer à **0.5** (standard COCO)
- Monitorer doublons: Si >5% → monter à 0.6
- Monitorer sous-détection: Si manque instances → descendre à 0.4

### 3. Moyenne Pondérée HV

**Pondération recommandée: Distance au bord (MVP)**

```python
def fuse_hv_maps_weighted(hv_A, hv_B, overlap_region):
    """
    Fusionne cartes HV dans zone overlap via moyenne pondérée.

    Args:
        hv_A, hv_B: Cartes HV (2, H, W) des 2 patches
        overlap_region: Coordonnées zone overlap (x1, y1, x2, y2)

    Returns:
        hv_fused: Carte HV lissée dans overlap
    """
    hv_fused = np.zeros_like(hv_A)

    for y in range(overlap_region.y1, overlap_region.y2):
        for x in range(overlap_region.x1, overlap_region.x2):
            # Distance au bord de chaque patch
            dist_A = min(x - patch_A.x1, patch_A.x2 - x,
                        y - patch_A.y1, patch_A.y2 - y)
            dist_B = min(x - patch_B.x1, patch_B.x2 - x,
                        y - patch_B.y1, patch_B.y2 - y)

            # Pondération normalisée
            weight_A = dist_A / (dist_A + dist_B)
            weight_B = dist_B / (dist_A + dist_B)

            # Moyenne pondérée
            hv_fused[:, y, x] = weight_A * hv_A[:, y, x] + weight_B * hv_B[:, y, x]

    return hv_fused
```

**Alternative optimisation: Pondération par confiance**

```python
# Au lieu de distance, utiliser confiance locale NP
conf_A = np_pred_A[y, x]
conf_B = np_pred_B[y, x]

weight_A = conf_A / (conf_A + conf_B + 1e-8)
weight_B = conf_B / (conf_A + conf_B + 1e-8)
```

### 4. Cas Particulier: Coins (4 Patches Overlap)

**Approche MVP: NMS Pairwise**

```python
# Fusionner séquentiellement
result = nms(nms(nms(patch_A, patch_B), patch_C), patch_D)
```

**Optimisation future: NMS Global**

```python
instances = [inst_A, inst_B, inst_C, inst_D]
keep = argmax([nms_score(i) for i in instances if IoU(i, others) > threshold])
```

---

## Risques Production

### Risque 1: Performance Temps Réel

**Calcul:**
- WSI 40,000×40,000: ~40,000 patches (stride 200px)
- Inférence 50ms/patch: **33 min/lame**

**Mitigations:**
- GPU parallélisation (4 GPUs → 8 min/lame)
- Optimisation TensorRT/ONNX (-30% temps)
- Stride adaptatif selon densité (-10% patches)

### Risque 2: RAM/VRAM Fusion

**Problème:**
- 40,000 patches × 600 KB = **24 GB RAM minimum**

**Solution: Streaming Stitching**

```python
# Ne pas charger toutes prédictions en RAM
# Fusionner ligne par ligne
for row_idx in range(n_rows):
    patches_row = predict_row(row_idx)
    fused_row = stitch_row(patches_row)
    save_to_disk(fused_row)
    del patches_row  # Libérer RAM
```

### Risque 3: Validation Qualité Stitching

**Métriques à suivre:**
1. Taux doublons détectés (instances fusionnées / total)
2. AJI avant/après stitching (gain attendu +5-10%)
3. Continuité aux jointures (gradients HV smooth?)

**Validation dataset:**
- Créer annotations sur **zones overlap spécifiquement**
- Mesurer erreurs aux jointures vs erreurs au centre patches

---

## Optimisations Futures

### 1. Stride Adaptatif

```python
density = estimate_nuclei_density(patch)

if density > threshold_high:
    stride = 200  # Tumeur dense → overlap max
elif density > threshold_medium:
    stride = 210  # Densité moyenne
else:
    stride = 220  # Stroma sparse → overlap min
```

**Gain:** -5-10% inférences

### 2. Confidence-Based Overlap

```python
uncertainty = compute_entropy(np_pred)

if uncertainty < 0.3:
    stride = 220  # Confiant → overlap minimal
else:
    stride = 180  # Incertain → overlap maximal
```

**Gain:** Adaptation automatique difficulté

### 3. Multi-Scale Fusion

```python
# Inférence à plusieurs résolutions
preds_high = model(patch_224)       # Haute résolution
preds_low = model(resize(patch_224, 112))  # Contexte large

# Fusion prédictions
final = 0.7 * preds_high + 0.3 * preds_low
```

**Gain:** Robustesse contexte spatial

---

## Timeline Implémentation (Post-MVP)

| Phase | Durée | Dépendances |
|-------|-------|-------------|
| Phase 1: NMS basique | 2 jours | Crops indépendants validés |
| Phase 2: HV pondéré | 1 jour | NMS fonctionnel |
| Phase 3: Streaming | 2 jours | RAM profiling |
| Phase 4: Validation | 3 jours | Annotations overlap |
| Phase 5: Optimisations | 5 jours | Métriques production |

**Total:** ~2-3 semaines développement + validation

---

## Références

- CoNIC Challenge (2022): Multi-resolution stitching strategies
- HoVer-Net (Graham 2019): Instance-level post-processing
- Mask R-CNN (He 2017): NMS for instance segmentation
- QuPath (Bankhead 2017): WSI tiling strategies

---

**Date création:** 2025-12-27
**Auteur:** Claude (session V13 Smart Crops validation)
**Status:** Archived for future reference
**Next review:** Après validation MVP crops indépendants
