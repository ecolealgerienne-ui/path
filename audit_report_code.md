# Rapport d'Audit - Code Preprocessing
**Date:** 2025-12-22
**Fichiers auditĂ©s:** 15

---

## 1. Constantes de Normalisation

### HOPTIMUS_MEAN
**Occurrences:** 11

❌ **INCOHÉRENT** - Valeurs différentes détectées!

**Valeur:** `HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)`
**Fichiers (10):**
- scripts/demo/gradio_demo.py
- scripts/evaluation/compare_train_vs_inference.py
- scripts/preprocessing/extract_features.py
- scripts/validation/diagnose_organ_prediction.py
- scripts/validation/test_organ_prediction_batch.py
- scripts/validation/verify_features.py
- src/inference/hoptimus_hovernet.py
- src/inference/hoptimus_unetr.py
- src/inference/optimus_gate_inference.py
- src/inference/optimus_gate_inference_multifamily.py

**Valeur:** `HOPTIMUS_MEAN = np.array([0.707223, 0.578729, 0.703617])`
**Fichiers (1):**
- scripts/preprocessing/extract_fold_features.py


### HOPTIMUS_STD
**Occurrences:** 11

❌ **INCOHÉRENT** - Valeurs différentes détectées!

**Valeur:** `HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)`
**Fichiers (10):**
- scripts/demo/gradio_demo.py
- scripts/evaluation/compare_train_vs_inference.py
- scripts/preprocessing/extract_features.py
- scripts/validation/diagnose_organ_prediction.py
- scripts/validation/test_organ_prediction_batch.py
- scripts/validation/verify_features.py
- src/inference/hoptimus_hovernet.py
- src/inference/hoptimus_unetr.py
- src/inference/optimus_gate_inference.py
- src/inference/optimus_gate_inference_multifamily.py

**Valeur:** `HOPTIMUS_STD = np.array([0.211883, 0.230117, 0.177517])`
**Fichiers (1):**
- scripts/preprocessing/extract_fold_features.py


## 2. Fonctions de Preprocessing

### `create_hoptimus_transform()`
**Implémentations trouvées:** 5

**Versions uniques:** 2

❌ **2 VERSIONS DIFFÉRENTES** détectées!

**Version b1b165da** (1 fichiers):
- scripts/validation/diagnose_organ_prediction.py

**Version 6e1c0f54** (4 fichiers):
- src/inference/hoptimus_hovernet.py
- src/inference/hoptimus_unetr.py
- src/inference/optimus_gate_inference.py
- src/inference/optimus_gate_inference_multifamily.py


### `preprocess()`
**Implémentations trouvées:** 6

**Versions uniques:** 3

❌ **3 VERSIONS DIFFÉRENTES** détectées!

**Version 4cc8a122** (1 fichiers):
- src/inference/cellvit_inference.py

**Version 00838da8** (1 fichiers):
- src/inference/cellvit_official.py

**Version 8cf13375** (4 fichiers):
- src/inference/hoptimus_hovernet.py
- src/inference/hoptimus_unetr.py
- src/inference/optimus_gate_inference.py
- src/inference/optimus_gate_inference_multifamily.py


## 3. Duplications Exactes

Fonctions dupliquées identiques (même code, plusieurs endroits):

### `create()` - 4 copies exactes

- src/inference/hoptimus_hovernet.py
- src/inference/hoptimus_unetr.py
- src/inference/optimus_gate_inference.py
- src/inference/optimus_gate_inference_multifamily.py

### `preprocess()` - 4 copies exactes

- src/inference/hoptimus_hovernet.py
- src/inference/hoptimus_unetr.py
- src/inference/optimus_gate_inference.py
- src/inference/optimus_gate_inference_multifamily.py


## 4. Recommandations

### Statistiques

- **Constantes dupliquées:** 22 occurrences
- **Fonctions dupliquées:** 11 implémentations
- **Duplications exactes:** 2 fonctions

### Actions Prioritaires

1. **Créer `src/preprocessing/__init__.py`**
   - Centraliser HOPTIMUS_MEAN, HOPTIMUS_STD
   - Centraliser create_hoptimus_transform()
   - Centraliser preprocess_image()

2. **Mettre à jour tous les fichiers**
   - Remplacer les constantes locales par imports
   - Remplacer les fonctions locales par imports

3. **Ajouter tests de cohérence**
   - Vérifier que tous les fichiers utilisent le même preprocessing

