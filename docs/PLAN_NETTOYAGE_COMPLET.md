# Plan de Nettoyage Complet du Projet

**Date:** 2025-12-24
**Statut:** Outils créés, prêt pour exécution
**Objectif:** Nettoyer scripts et données avant re-train sur bases saines

---

## Contexte

Suite à la découverte du Bug #6 (feature path mismatch), nous avons:

1. ✅ Identifié la cause racine (argument default incorrect)
2. ✅ Fixé le code (`train_hovernet_family.py` ligne 333)
3. ✅ Créé diagnostic confirmant le problème

**Mais avant le re-train**, l'utilisateur a demandé de **nettoyer complètement** le projet:

> "Avant d'aller plus loin, il faut nettoyer les scripts pour pointer sur les mêmes données, ensuite il faut nettoyer les data avant, après on lance le train sur des bonnes bases."

---

## Outils Créés

### 1. `scripts/utils/audit_project_paths.py`

**Rôle:** Audit complet du projet

**Vérifie:**
- ✅ Tous les scripts Python et leurs références de paths
- ✅ Utilisation des constantes vs paths hardcodés
- ✅ Données existantes sur disque (taille, timestamps)
- ✅ Identifie redondances et incohérences

**Sortie:**
- Rapport console détaillé
- Fichier JSON `results/audit_project_paths.json`
- Plan de nettoyage avec actions recommandées

**Usage:**
```bash
python scripts/utils/audit_project_paths.py
```

---

### 2. `scripts/utils/cleanup_project_data.py`

**Rôle:** Nettoyage automatisé des données

**Supprime:**
- `data/cache/family_data/` (redondant avec `data/family_data/`)
- `data/cache/pannuke_features/` (features OLD avec Bug #1 et #2)

**Conserve:**
- `data/family_data/` (source de vérité, CLS std 0.770)
- `data/family_FIXED/` (fichiers source originaux)

**Options:**
- `--dry-run` : Voir ce qui sera supprimé sans supprimer
- `--backup` : Créer sauvegardes avant suppression
- `--force` : Supprimer sans confirmation

**Usage:**
```bash
# Voir ce qui sera supprimé
python scripts/utils/cleanup_project_data.py --dry-run

# Exécuter avec sauvegarde
python scripts/utils/cleanup_project_data.py --backup

# Exécuter direct (ATTENTION: irréversible!)
python scripts/utils/cleanup_project_data.py --force
```

---

### 3. `scripts/utils/prepare_clean_training.sh`

**Rôle:** Orchestration complète du processus

**Étapes automatisées:**
1. Audit complet du projet
2. Migration scripts vers constantes (si nécessaire)
3. Nettoyage données redondantes
4. Validation données essentielles
5. Diagnostic path mismatch final

**Usage:**
```bash
# Dry-run complet
bash scripts/utils/prepare_clean_training.sh --dry-run

# Exécution réelle
bash scripts/utils/prepare_clean_training.sh
```

---

## Plan d'Exécution Recommandé

### Phase 1: Audit (2 min)

```bash
python scripts/utils/audit_project_paths.py
```

**Vérifiez dans la sortie:**
- Nombre de scripts avec paths hardcodés
- Taille totale des données redondantes
- Plan de nettoyage proposé

---

### Phase 2: Nettoyage Dry-Run (1 min)

```bash
python scripts/utils/cleanup_project_data.py --dry-run
```

**Vérifiez dans la sortie:**
- Quels répertoires seront supprimés
- Espace disque qui sera libéré
- Que `data/family_data/` est conservé

---

### Phase 3: Nettoyage Réel (1 min)

**Option A (Recommandée): Avec sauvegarde**
```bash
python scripts/utils/cleanup_project_data.py --backup
```

**Option B (Rapide): Sans sauvegarde**
```bash
python scripts/utils/cleanup_project_data.py --force
```

**Résultat attendu:**
```
✅ Supprimés: 2 répertoires
💾 Espace libéré: ~1.5 GB
```

---

### Phase 4: Validation (1 min)

```bash
python scripts/validation/diagnose_training_data_mismatch.py --family epidermal
```

**Sortie attendue:**
```
✅ CONFIGURATION CORRECTE:
  - Features trouvées dans data/family_data
  - CLS std: 0.770
  - Prêt pour re-train
```

---

### Phase 5: Re-Train (40 min)

```bash
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment
```

**Résultat attendu:**
```
Epoch 50/50:
  NP Dice:  0.9500 ± 0.0050
  HV MSE:   0.0150 ± 0.0020
  NT Acc:   0.9000 ± 0.0100
```

---

### Phase 6: Test (5 min)

**Test sur training data:**
```bash
python scripts/validation/test_on_training_data.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 10
```

**Résultat attendu:**
```
NP Dice:  0.9500 ± 0.0050  ✅ (au lieu de 0.0000)
HV MSE:   0.0150 ± 0.0020  ✅
NT Acc:   0.9000 ± 0.0100  ✅
```

**Test AJI:**
```bash
python scripts/evaluation/test_aji_v8.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 50
```

**Résultat attendu:**
```
AJI:  0.6000 ± 0.0500  ✅ (objectif atteint!)
```

---

## Timeline Complète

| Étape | Durée | Commande |
|-------|-------|----------|
| 1. Audit | 2 min | `audit_project_paths.py` |
| 2. Dry-run | 1 min | `cleanup_project_data.py --dry-run` |
| 3. Nettoyage | 1 min | `cleanup_project_data.py --backup` |
| 4. Validation | 1 min | `diagnose_training_data_mismatch.py` |
| 5. Re-train | 40 min | `train_hovernet_family.py` |
| 6. Test | 5 min | `test_on_training_data.py` + `test_aji_v8.py` |
| **TOTAL** | **50 min** | |

---

## Script Master (Option Rapide)

**Pour exécuter toute la préparation en une commande:**

```bash
bash scripts/utils/prepare_clean_training.sh
```

Ce script orchestre:
1. Audit
2. Migration scripts (si nécessaire)
3. Nettoyage données
4. Validation
5. Diagnostic final

Puis affiche les commandes de re-train à lancer manuellement.

---

## Données Conservées vs Supprimées

### ✅ CONSERVÉ (Source de Vérité)

**`data/family_data/`** (852.4 MB)
```
epidermal_features.npz   # CLS std 0.770 ✅ CORRECT
epidermal_targets.npz
```

**`data/family_FIXED/`** (177.4 MB)
```
epidermal_data_FIXED.npz  # Source originale
```

---

### ❌ SUPPRIMÉ (Redondant/Obsolète)

**`data/cache/family_data/`** (~500 MB si existe)
- Raison: Redondant avec `data/family_data/`
- Potentiellement obsolète (timestamps plus anciens)

**`data/cache/pannuke_features/`** (~12 GB si existe)
- Raison: Features extraites AVANT fix Bug #1 et #2
- CLS std incorrect (~0.82 au lieu de 0.77)
- Normalisation potentiellement corrompue

---

## Vérifications de Sécurité

Avant de supprimer quoi que ce soit, le script vérifie:

1. ✅ `data/family_data/` existe et contient features/targets
2. ✅ CLS std dans plage correcte [0.70-0.90]
3. ✅ `data/family_FIXED/` existe (source de secours)

**Si une de ces vérifications échoue:** Le script s'arrête et affiche:
```
❌ ERREUR CRITIQUE: Données essentielles manquantes!
   Lancez: python scripts/preprocessing/extract_features_from_fixed.py --family epidermal
```

---

## Avantages du Nettoyage

### 1. Espace Disque

**Avant:**
```
data/family_data/         ~850 MB
data/cache/family_data/   ~500 MB (si existe)
data/cache/pannuke_features/  ~12 GB (si existe)
TOTAL:                    ~13.5 GB
```

**Après:**
```
data/family_data/         ~850 MB  ✅ Source de vérité
data/family_FIXED/        ~177 MB  ✅ Source originale
TOTAL:                    ~1 GB
```

**Gain:** ~12.5 GB libérés

---

### 2. Cohérence Garantie

**Avant:**
- Scripts utilisaient différents chemins
- Risque de mismatch training/test
- Confusion sur "quelle version des données?"

**Après:**
- ✅ Une seule source de vérité: `data/family_data/`
- ✅ Tous les scripts utilisent `DEFAULT_FAMILY_DATA_DIR`
- ✅ Impossible d'avoir du mismatch

---

### 3. Performances

**Avant:**
- Modèle entraîné sur features ?
- Test sur features différentes ?
- NP Dice 0.0000 (OOD)

**Après:**
- ✅ Training et test sur MÊMES features
- ✅ CLS std validé (0.770)
- ✅ NP Dice attendu ~0.95

---

## Rollback (Si Problème)

Si le nettoyage cause un problème, vous pouvez restaurer:

**Avec `--backup`:**
```bash
# Les sauvegardes sont dans:
ls data/backups/

# Restaurer:
mv data/backups/family_data_BACKUP_20251224_* data/cache/family_data
```

**Sans backup:**
```bash
# Ré-extraire features depuis FIXED:
python scripts/preprocessing/extract_features_from_fixed.py --family epidermal
```

---

## Prochaines Étapes (Après Nettoyage)

1. **Si test epidermal réussit (Dice ~0.95, AJI >0.60):**
   ```bash
   # Re-train les 4 autres familles (10h total)
   for family in glandular digestive urologic respiratory; do
       python scripts/training/train_hovernet_family.py \
           --family $family \
           --epochs 50 \
           --augment
   done
   ```

2. **Si test epidermal échoue encore:**
   - Vérifier CLS std avec `verify_features.py`
   - Inspecter visuel avec `visualize_raw_predictions.py`
   - Créer issue GitHub avec logs complets

---

## Références

- **Bug #6 Documentation:** `docs/BUG_6_FEATURE_PATH_MISMATCH.md`
- **Plan de résolution:** `docs/BUG_6_RESOLUTION_PLAN.md`
- **Diagnostic utilisé:** `scripts/validation/diagnose_training_data_mismatch.py`
- **Scripts créés:** `scripts/utils/audit_project_paths.py`, `cleanup_project_data.py`, `prepare_clean_training.sh`
