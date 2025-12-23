# Plan de Vérification Méthodique : HoVer-Net vs Notre Système

**Date** : 2025-12-23
**Objectif** : Identifier EXACTEMENT pourquoi notre système (entraîné sur PanNuke) n'obtient pas les résultats HoVer-Net baseline
**Méthodologie** : Comparaison systématique INPUT → TRAINING → MODEL → OUTPUT

---

## ❓ Question Centrale

> **HoVer-Net original (Graham et al. 2019) entraîné sur PanNuke → AJI ~0.68**
> **Notre système (HoVer-Net sur H-optimus-0) entraîné sur PanNuke → AJI 0.0863**
>
> Différence : **8× pire** — Pourquoi ?

---

## 🔍 Étape 1 : Vérifier les Données d'Entraînement UTILISÉES

### Objectif
Déterminer **quelles données ont réellement été utilisées** pour entraîner les modèles actuels.

### Tests à Réaliser

#### Test 1.1 : Chercher les données d'entraînement existantes
```bash
# Chercher TOUS les fichiers de données par famille
find . -name "*glandular*" -o -name "*digestive*" -o -name "*urologic*" | grep -E "\.(npz|npy)$"

# Vérifier les timestamps (quand ont-ils été créés ?)
ls -lh --time-style=full-iso <fichiers_trouvés>
```

**Questions à répondre** :
- [ ] Des fichiers `*_data.npz` existent-ils ?
- [ ] Des fichiers `*_data_FIXED.npz` existent-ils ?
- [ ] Quelles sont leurs dates de création ?
- [ ] Quelle est leur taille (cohérente avec nb d'échantillons attendus) ?

#### Test 1.2 : Inspecter le format des instances dans les données
```bash
# Script à créer : inspect_training_instances.py
# Charge un fichier .npz et affiche :
# - Nombre d'instances par image
# - Distribution des tailles d'instances
# - Exemple de inst_map avec IDs
```

**Questions à répondre** :
- [ ] Les inst_map contiennent-ils des IDs séquentiels (1, 2, 3...) ou des IDs PanNuke natifs ?
- [ ] Y a-t-il des instances fusionnées (grandes blobs) ou des instances séparées (petits blobs) ?
- [ ] Combien d'instances par image en moyenne ?

**Critère de validation** :
- ✅ Si inst_map a 50-100 instances/image → Probablement FIXED (vraies instances)
- ❌ Si inst_map a 5-15 instances/image → Probablement OLD (connectedComponents fusionnées)

---

## 🔍 Étape 2 : Comparer Preprocessing PanNuke (HoVer-Net vs Nous)

### Objectif
Vérifier si notre preprocessing PanNuke est **identique** à HoVer-Net original.

### Tests à Réaliser

#### Test 2.1 : Lire le paper HoVer-Net (Graham et al. 2019)
- [ ] Section "Dataset" : Comment PanNuke est-il prétraité ?
- [ ] Utilisent-ils les IDs natifs ou connectedComponents ?
- [ ] Quelle est la distribution d'instances par image rapportée ?

**Ressource** :
- Paper : "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images"
- Lien probable : https://arxiv.org/abs/1812.06499

#### Test 2.2 : Comparer avec leur code officiel
- [ ] Repo GitHub : https://github.com/vqdang/hover_net
- [ ] Inspecter `process.py` ou équivalent : Comment extraient-ils les instances PanNuke ?
- [ ] Comparer avec notre `prepare_family_data.py` ligne par ligne

**Questions à répondre** :
- [ ] HoVer-Net utilise-t-il `connectedComponents` ou IDs natifs ?
- [ ] Quel est le format exact de leurs HV targets ?
- [ ] Y a-t-il des différences de normalisation [-1, 1] ?

**Critère de validation** :
- ✅ Si HoVer-Net utilise aussi connectedComponents → Notre OLD data est correcte
- ❌ Si HoVer-Net utilise IDs natifs → Nous devons utiliser FIXED data

---

## 🔍 Étape 3 : Comparer Architecture Modèle

### Objectif
Vérifier si notre décodeur HoVer-Net est **identique** à l'original.

### Tests à Réaliser

#### Test 3.1 : Comparer architectures
```python
# Notre décodeur : src/models/hovernet_decoder.py
# HoVer-Net original : hover_net/models/hovernet/net_desc.py (repo GitHub)
```

**Questions à répondre** :
- [ ] Nombre de couches identique ?
- [ ] Skip connections identiques ?
- [ ] Fonctions d'activation (ReLU vs autre) ?
- [ ] Poids de loss (λ_np, λ_hv, λ_nt) identiques ?

#### Test 3.2 : Vérifier la loss function
```python
# HoVer-Net original : MSE pour HV, CrossEntropy pour NP/NT
# Notre implémentation : SmoothL1Loss pour HV (depuis 2025-12-20)
```

**Différence critique identifiée** :
- HoVer-Net original : `MSE` pour HV
- Notre système : `SmoothL1Loss` pour HV (moins sensible aux outliers)

**Question** : Est-ce que SmoothL1Loss peut causer des gradients plus faibles ?

**Test à faire** :
- [ ] Comparer MSE vs SmoothL1Loss sur un batch
- [ ] Ré-entraîner UNE famille avec MSE pour comparer

---

## 🔍 Étape 4 : Comparer Post-Processing Watershed

### Objectif
Vérifier si notre implémentation watershed est **identique** à HoVer-Net original.

### Tests à Réaliser

#### Test 4.1 : Comparer implémentations watershed
```python
# HoVer-Net original : hover_net/infer/post_proc.py
# Notre système : src/inference/optimus_gate_inference_multifamily.py (méthode watershed)
```

**Questions à répondre** :
- [ ] Mêmes seuils `edge_threshold` ? (notre 0.3 vs leur ?)
- [ ] Mêmes seuils `dist_threshold` ? (notre 2 vs leur ?)
- [ ] Même algorithme de détection de markers ?
- [ ] Utilisent-ils un pré-traitement des HV maps (smoothing, etc.) ?

#### Test 4.2 : Tester avec leurs paramètres exacts
```bash
# Une fois les paramètres HoVer-Net identifiés, tester sur nos données
python scripts/evaluation/test_watershed_params.py \
    --edge_threshold <valeur_hovernet> \
    --dist_threshold <valeur_hovernet>
```

---

## 🔍 Étape 5 : Reproduire HoVer-Net Baseline sur PanNuke

### Objectif
**Preuve ultime** : Reproduire exactement les résultats HoVer-Net paper.

### Tests à Réaliser

#### Test 5.1 : Utiliser le modèle HoVer-Net pré-entraîné officiel
```bash
# Télécharger leur checkpoint pré-entraîné
# Source : https://github.com/vqdang/hover_net (releases)

# Évaluer sur notre subset PanNuke fold2
python hover_net/run_infer.py \
    --checkpoint hovernet_pannuke_official.pth \
    --input_dir data/evaluation/pannuke_fold2_converted
```

**Questions à répondre** :
- [ ] Quel AJI obtiennent-ils sur fold2 ?
- [ ] Combien d'instances détectées par image en moyenne ?
- [ ] Recall/Precision comparés aux nôtres ?

**Critère de validation** :
- ✅ Si leur modèle obtient aussi AJI ~0.09 sur fold2 → Problème dans les données GT
- ❌ Si leur modèle obtient AJI ~0.60-0.70 → Notre implémentation a un bug

---

## 📊 Matrice de Diagnostic

| Étape | Test | Résultat Attendu | Action si ❌ |
|-------|------|------------------|--------------|
| 1.1 | Données utilisées | Fichiers *_data.npz trouvés | Générer données manquantes |
| 1.2 | Format instances | 50-100 inst/image (FIXED) | Vérifier connectedComponents vs natif |
| 2.1 | Paper HoVer-Net | Méthode extraction instances | Comparer avec notre script |
| 2.2 | Code GitHub HoVer-Net | Ligne par ligne identique | Corriger différences |
| 3.1 | Architecture | Décodeur identique | Ajuster couches/activations |
| 3.2 | Loss function | MSE vs SmoothL1Loss | Tester avec MSE |
| 4.1 | Watershed params | Seuils identiques | Ajuster nos seuils |
| 5.1 | Modèle officiel | AJI ~0.60-0.70 | Reproduire leur pipeline |

---

## 🎯 Critères de Décision APRÈS Tests

### Scénario A : Preprocessing Différent
**Si** : HoVer-Net utilise IDs natifs ET nous utilisons connectedComponents
**Action** : Générer données FIXED + ré-entraîner (10h)
**Gain estimé** : AJI 0.09 → 0.60-0.70

### Scénario B : Architecture/Loss Différente
**Si** : SmoothL1Loss cause gradients faibles vs MSE
**Action** : Ré-entraîner UNE famille avec MSE (2h test)
**Gain estimé** : AJI 0.09 → 0.30-0.40 (si confirmé)

### Scénario C : Watershed Différent
**Si** : Paramètres watershed très différents
**Action** : Ajuster paramètres (1h)
**Gain estimé** : AJI 0.09 → 0.15-0.25

### Scénario D : Combination
**Si** : Plusieurs différences identifiées
**Action** : Corriger dans l'ordre : Preprocessing → Architecture → Watershed
**Gain estimé** : AJI 0.09 → 0.70-0.80 (cumulatif)

---

## 📝 Scripts à Créer

### Script 1 : `inspect_training_instances.py`
```python
"""Inspecte les instances dans les fichiers de données d'entraînement."""
# Charge un .npz
# Affiche nombre d'instances par image
# Visualise quelques inst_map
```

### Script 2 : `compare_hovernet_preprocessing.py`
```python
"""Compare notre preprocessing avec HoVer-Net officiel."""
# Lit le même batch PanNuke
# Applique les deux pipelines
# Compare les inst_map résultants
```

### Script 3 : `test_loss_functions.py`
```python
"""Compare MSE vs SmoothL1Loss sur un batch."""
# Charge un batch
# Calcule loss avec les deux méthodes
# Compare magnitudes de gradients
```

---

## ⏱️ Timeline Estimée

| Étape | Temps | Dépendances |
|-------|-------|-------------|
| 1. Vérifier données | 30 min | Accès filesystem |
| 2. Lire paper + code | 2h | Internet, lecture |
| 3. Comparer architecture | 1h | Code review |
| 4. Comparer watershed | 1h | Code review |
| 5. Tester modèle officiel | 1h | Téléchargement checkpoint |
| **TOTAL** | **5.5h** | Avant toute décision |

---

## ✅ Checklist de Validation

Avant de proposer TOUTE solution, vérifier :

- [ ] **Étape 1 complète** : Nous savons quelles données ont été utilisées
- [ ] **Étape 2 complète** : Nous avons comparé avec HoVer-Net preprocessing
- [ ] **Étape 3 complète** : Nous avons comparé architecture et loss
- [ ] **Étape 4 complète** : Nous avons comparé watershed
- [ ] **Étape 5 complète** : Nous avons testé leur modèle officiel
- [ ] **Rapport écrit** : Différences identifiées documentées
- [ ] **Consensus** : Solution validée avec l'utilisateur

---

## 🚫 INTERDICTIONS

- ❌ NE PAS générer de nouvelles données avant Étape 1 et 2
- ❌ NE PAS ré-entraîner avant Étape 3
- ❌ NE PAS modifier watershed avant Étape 4
- ❌ NE PAS proposer de solution avant Étape 5

**Principe** : COMPRENDRE avant AGIR
