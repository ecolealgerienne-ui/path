# Situation Environnement — 2025-12-22

**Date:** 2025-12-22 14:08
**Contexte:** Tentative d'exécution du pipeline de validation par famille

---

## 🔴 BLOCAGE IDENTIFIÉ

Le pipeline de validation ne peut pas s'exécuter car l'environnement actuel ne contient ni les données ni les modèles entraînés.

### Diagnostic Complet

```bash
# Données PanNuke
❌ /home/amar/data/PanNuke/ → Directory does not exist
❌ ./data/ → Directory does not exist
❌ Features pré-extraites (.npz) → Aucun fichier trouvé

# Checkpoints entraînés
❌ models/checkpoints/*.pth → No such file or directory
✅ models/ → Existe (vide sauf models/pretrained/)

# Scripts validés (Phase 1)
✅ scripts/evaluation/prepare_test_samples_by_family.py
✅ scripts/evaluation/test_family_models_isolated.py
✅ scripts/evaluation/test_organ_routing.py
✅ scripts/evaluation/run_family_validation_pipeline.sh
```

---

## 📊 Ce Qui Existe

| Élément | Statut | Détails |
|---------|--------|---------|
| **Code source** | ✅ Complet | src/, scripts/, tests/ |
| **Documentation** | ✅ À jour | CLAUDE.md, docs/, guides |
| **Scripts validation** | ✅ Prêts | Tous conformes Phase 1 |
| **Données PanNuke** | ❌ Manquantes | Aucun fichier .npy trouvé |
| **Features extraites** | ❌ Manquantes | Aucun .npz trouvé |
| **Checkpoints HoVer-Net** | ❌ Manquants | 5 familles à ré-entraîner |
| **Checkpoint OrganHead** | ❌ Manquant | À ré-entraîner |

---

## 🛠️ Solutions Possibles

### Option 1: Setup Complet (Environnement Local)

**Durée estimée:** 12-24 heures (téléchargement + entraînement)

#### Étape 1: Télécharger PanNuke (~1.5 GB)

```bash
# Créer répertoire de destination
mkdir -p /home/amar/data

# Télécharger et préparer PanNuke
python scripts/setup/download_and_prepare_pannuke.py \
    --output_dir /home/amar/data/PanNuke

# Vérifier structure
ls -la /home/amar/data/PanNuke/fold*/
```

**Structure attendue:**
```
/home/amar/data/PanNuke/
├── fold0/
│   ├── images.npy  # (2656, 256, 256, 3) uint8
│   ├── masks.npy   # (2656, 256, 256, 6) uint8
│   └── types.npy   # (2656,) str
├── fold1/ (idem)
└── fold2/ (idem)
```

#### Étape 2: Extraire Features H-optimus-0 (~2-3 heures)

```bash
# Extraire features pour les 3 folds (avec chunking pour économiser RAM)
for fold in 0 1 2; do
    python scripts/preprocessing/extract_features.py \
        --data_dir /home/amar/data/PanNuke \
        --fold $fold \
        --batch_size 8 \
        --chunk_size 500
done

# Vérifier qualité (CLS std doit être 0.70-0.90)
python scripts/validation/verify_features.py \
    --features_dir data/cache/pannuke_features
```

#### Étape 3: Entraîner OrganHead (~10 minutes)

```bash
python scripts/training/train_organ_head.py \
    --folds 0 1 2 \
    --epochs 50

# Résultat attendu: Val Accuracy ~99.94%
```

#### Étape 4: Entraîner 5 Familles HoVer-Net (~5-10 heures)

```bash
for family in glandular digestive urologic epidermal respiratory; do
    python scripts/training/train_hovernet_family.py \
        --family $family \
        --epochs 50 \
        --augment
done
```

**Résultats attendus:**
| Famille | NP Dice | HV MSE | NT Acc |
|---------|---------|--------|--------|
| Glandular | 0.9648 | 0.0106 | 0.9111 |
| Digestive | 0.9634 | 0.0163 | 0.8824 |
| Urologic | 0.9318 | 0.2812 | 0.9139 |
| Epidermal | 0.9542 | 0.2653 | 0.8857 |
| Respiratory | 0.9409 | 0.0500 | 0.9183 |

#### Étape 5: Exécuter Pipeline Validation

```bash
bash scripts/evaluation/run_family_validation_pipeline.sh \
    /home/amar/data/PanNuke \
    models/checkpoints
```

---

### Option 2: Transfert depuis Machine avec Données

Si les données et checkpoints existent ailleurs:

```bash
# Sur la machine source
tar -czf cellvit_data.tar.gz /path/to/PanNuke
tar -czf cellvit_checkpoints.tar.gz /path/to/models/checkpoints

# Transfert
scp cellvit_data.tar.gz user@target:/home/amar/data/
scp cellvit_checkpoints.tar.gz user@target:/path/to/project/models/

# Sur la machine cible
cd /home/amar/data
tar -xzf cellvit_data.tar.gz

cd /path/to/project/models
tar -xzf cellvit_checkpoints.tar.gz
```

---

### Option 3: Utiliser Environnement Cloud/Serveur

Si un serveur avec GPU contient déjà tout:

```bash
# SSH vers serveur
ssh user@server.domain

# Naviguer vers projet
cd /path/to/cellvit-optimus

# Exécuter pipeline
bash scripts/evaluation/run_family_validation_pipeline.sh \
    /data/PanNuke \
    models/checkpoints
```

---

## 🎯 Recommandation

**Pour diagnostic rapide (prochaine session):**
- Utiliser **Option 2 ou 3** si possible (gain de temps)
- Sinon, **Option 1** mais prévoir 12-24h de setup

**Vérification avant exécution:**
```bash
# Checklist rapide
[ -d /home/amar/data/PanNuke/fold2 ] && echo "✅ PanNuke OK" || echo "❌ PanNuke manquant"
[ -f models/checkpoints/organ_head_best.pth ] && echo "✅ OrganHead OK" || echo "❌ OrganHead manquant"
[ -f models/checkpoints/hovernet_glandular_best.pth ] && echo "✅ HoVer-Net OK" || echo "❌ HoVer-Net manquants"
```

---

## 📝 État Actuel du Code

Tout le code est prêt et validé:
- ✅ Scripts conformes Phase 1 (modules centralisés)
- ✅ Bugs de compatibilité corrigés (num_classes, strict=False)
- ✅ Optimisation mémoire (mmap, chunking)
- ✅ Pipeline orchestré et documenté

**Il ne manque QUE les données et les modèles entraînés.**

---

## 🔄 Prochaines Étapes (Une Fois Données Disponibles)

1. Exécuter pipeline validation → Identifier scénario (1, 2 ou 3)
2. Analyser rapports JSON (isolated_tests/, routing_tests/)
3. Appliquer solution ciblée selon diagnostic
4. Documenter résultats dans CLAUDE.md

---

**Dernière mise à jour:** 2025-12-22 14:08
**Auteur:** Claude (Phase 1 Refactoring Complete)
