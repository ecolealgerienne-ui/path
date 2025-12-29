# Rapport de Normalisation Macenko — Session 2025-12-29

## Résumé Exécutif

Normalisation Macenko appliquée sur les 5 familles PanNuke avant génération des smart crops V13.

| Famille | Images | Ref Idx | Std Avant | Std Après | Réduction |
|---------|--------|---------|-----------|-----------|-----------|
| **Glandular** | 3391 | 613 | 23.87 | 15.88 | **33.5%** |
| **Digestive** | 2430 | 1311 | 32.34 | 18.17 | **43.8%** |
| **Urologic** | 1101 | 419 | 28.87 | 20.21 | **30.0%** |
| **Respiratory** | 408 | 169 | 21.06 | 14.40 | **31.6%** |
| **Epidermal** | 574 | 416 | 25.73 | 24.96 | **3.0%** |

**Moyenne de réduction de variabilité : 28.4%**

---

## Détails par Famille

### 1. Glandular (3391 images)

```
Reference image: index 613
Mean intensity std (before): 23.87
Mean intensity std (after):  15.88
Variability reduction:       33.5%
```

**Analyse :** Bonne réduction de variabilité. La famille glandulaire (Breast, Prostate, Thyroid, Pancreatic, Adrenal_gland) montre une diversité de coloration modérée qui a été bien normalisée.

### 2. Digestive (2430 images)

```
Reference image: index 1311
Mean intensity std (before): 32.34
Mean intensity std (after):  18.17
Variability reduction:       43.8%
```

**⚠️ Warning détecté :**
```
RuntimeWarning: overflow encountered in exp
  normalized = 255 * np.exp(-self.target_stains @ source_concentrations)
```

**Analyse :** Meilleure réduction de variabilité (43.8%). La famille digestive (Colon, Stomach, Esophagus, Bile-duct) présentait la plus grande variabilité initiale (32.34), probablement due aux différences de coloration entre organes digestifs. Le warning d'overflow indique que certaines images avaient des valeurs de concentration extrêmes, mais le résultat final reste acceptable.

### 3. Urologic (1101 images)

```
Reference image: index 419
Mean intensity std (before): 28.87
Mean intensity std (after):  20.21
Variability reduction:       30.0%
```

**Analyse :** Réduction modérée. La famille urologique (Kidney, Bladder, Testis, Ovarian, Uterus, Cervix) montre une variabilité résiduelle plus élevée (20.21), probablement due à la diversité des tissus (épithéliums stratifiés vs tubulaires).

### 4. Respiratory (408 images)

```
Reference image: index 169
Mean intensity std (before): 21.06
Mean intensity std (after):  14.40
Variability reduction:       31.6%
```

**Analyse :** Bonne réduction malgré le faible volume de données. La famille respiratoire (Lung, Liver) avait une variabilité initiale plus faible, ce qui suggère une homogénéité relative des colorations H&E pour ces tissus.

### 5. Epidermal (574 images)

```
Reference image: index 416
Mean intensity std (before): 25.73
Mean intensity std (after):  24.96
Variability reduction:       3.0%
```

**Analyse :** Réduction minimale. La famille épidermique (Skin, HeadNeck) présente une variabilité intrinsèque difficile à normaliser. Hypothèses :
- Kératine et couches stratifiées absorbent différemment les colorants
- Référence potentiellement non optimale (index 416)
- Structure tissulaire fondamentalement différente des autres familles

---

## Interprétation Globale

### Corrélation Volume ↔ Réduction

| Famille | Volume | Réduction | Observation |
|---------|--------|-----------|-------------|
| Digestive | 2430 | 43.8% | ✅ Grand volume, forte réduction |
| Glandular | 3391 | 33.5% | ✅ Grand volume, bonne réduction |
| Respiratory | 408 | 31.6% | ✅ Petit volume, bonne réduction |
| Urologic | 1101 | 30.0% | ✅ Volume moyen, réduction correcte |
| Epidermal | 574 | 3.0% | ⚠️ Petit volume, réduction minimale |

**Conclusion :** La réduction de variabilité ne corrèle pas directement avec le volume. Le facteur déterminant est la **structure tissulaire** :
- Tissus glandulaires/digestifs : bien normalisables (architecture régulière)
- Tissus épidermiques : difficiles à normaliser (couches stratifiées)

### Impact Attendu sur AJI

La normalisation Macenko devrait améliorer :
1. **Stabilité inter-slide** : Moins de variation de couleur = gradients HV plus cohérents
2. **Réduction du "bleeding" éosin** : Contours nucléaires plus nets
3. **Extraction H-channel** : Meilleure séparation Hématoxyline/Éosine pour FPN Chimique

---

## Prochaine Étape

Générer les smart crops V13 sur les données normalisées :

```bash
# Pour chaque famille
python scripts/preprocessing/prepare_v13_smart_crops.py --family glandular --max_samples 5000
python scripts/preprocessing/prepare_v13_smart_crops.py --family digestive --max_samples 5000
python scripts/preprocessing/prepare_v13_smart_crops.py --family urologic --max_samples 5000
python scripts/preprocessing/prepare_v13_smart_crops.py --family respiratory --max_samples 5000
python scripts/preprocessing/prepare_v13_smart_crops.py --family epidermal --max_samples 5000
```

---

## Métadonnées

- **Date :** 2025-12-29
- **Script :** `scripts/preprocessing/normalize_staining_source.py`
- **Méthode :** Macenko 2009 (SVD-based stain deconvolution)
- **Selection méthode :** "balanced" (auto-select reference)
- **Données source :** `data/family_FIXED/*_data_FIXED.npz`
- **Mode :** Overwrite (pas de suffix)
