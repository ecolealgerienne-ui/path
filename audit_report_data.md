# Rapport d'Audit - Données
**Date:** 2025-12-22

---

## 1. Vue d'Ensemble

**Répertoires existants:** 2/13
**Espace disque total:** 19.78 KB


## 2. Détails par Répertoire

### `results`

**Chemin:** `results`
**Taille:** 19.78 KB (20,257 bytes)
**Fichiers:** 2

**Types de fichiers:**

| Extension | Nombre | Taille Totale |
|-----------|--------|---------------|
| `.md` | 2 | 19.78 KB |


### `data_root` ❌ N'existe pas


### `cache` ❌ N'existe pas


### `pannuke_features` ❌ N'existe pas


### `family_data` ❌ N'existe pas


### `family_data_FIXED` ❌ N'existe pas


### `evaluation` ❌ N'existe pas


### `samples` ❌ N'existe pas


### `snapshots` ❌ N'existe pas


### `feedback` ❌ N'existe pas


### `models_pretrained`

**Chemin:** `models/pretrained`
**Taille:** 0.00 B (0 bytes)
**Fichiers:** 1

**Types de fichiers:**

| Extension | Nombre | Taille Totale |
|-----------|--------|---------------|
| `no_ext` | 1 | 0.00 B |


### `models_checkpoints` ❌ N'existe pas


### `models_checkpoints_FIXED` ❌ N'existe pas


## 3. Analyse des Duplications

### Répertoires Suspects de Duplication


## 4. Recommandations de Nettoyage

### Actions Immédiates

1. **Supprimer les duplications validées**
   ```bash
   # Si family_FIXED est validé
   rm -rf data/family_data
   mv data/family_FIXED data/family_data

   # Si checkpoints_FIXED est validé
   rm -rf models/checkpoints
   mv models/checkpoints_FIXED models/checkpoints
   ```

2. **Centraliser les données pré-extraites**
   - Créer `data/preprocessed/` pour TOUTES les features H-optimus-0
   - Structure:
     ```
     data/preprocessed/
     ├── pannuke_features/  ← Features H-optimus-0
     ├── family_data/       ← Targets NP/HV/NT par famille
     └── metadata.json      ← Versions, hashes, dates
     ```

3. **Versioning des données**
   - Ajouter `metadata.json` dans chaque cache:
     ```json
     {
       "version": "2025-12-21-FIXED",
       "backbone": "H-optimus-0",
       "preprocessing": "forward_features_with_layernorm",
       "created_at": "2025-12-21T10:30:00",
       "num_samples": 7900,
       "hash": "a1b2c3d4"
     }
     ```

4. **Pipeline de génération unique**
   - Script `scripts/preprocessing/generate_all_data.py`:
     1. Extrait features H-optimus-0 (une fois)
     2. Génère family_data (une fois)
     3. Sauvegarde metadata
     4. Tous les scripts utilisent ces données

### Estimation d'Économie d'Espace

