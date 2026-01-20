# V14 Cytologie — Ordre d'Exécution du Pipeline

> **Date:** 2026-01-19
> **Clarification Architecturale Critique**

---

## ⚠️ RECTIFICATION IMPORTANTE

**Confusion Initiale:** Documentation précédente suggérait "parallélisme pur" entre CellPose et H-Optimus.

**Réalité:** Le pipeline est **Séquentiel PUIS Parallèle**.

---

## 🔄 Architecture Réelle (5 Étapes)

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PIPELINE V14 CYTOLOGIE (CORRECT)                    │
└─────────────────────────────────────────────────────────────────────┘

INPUT: Grande tuile WSI (ex: 1024×1024 pixels, ~50 cellules)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PHASE SÉQUENTIELLE (Obligatoire — Cannot Skip)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────┐
│ ÉTAPE 1: Le "Découpeur" (CellPose Master — nuclei)                  │
├─────────────────────────────────────────────────────────────────────┤
│  Action:  CellPose scanne la tuile 1024×1024                        │
│  Sortie:  • N bounding boxes (coordonnées x, y, w, h)               │
│           • N masques (contours exacts des noyaux)                   │
│           Exemple: Trouve 50 noyaux                                  │
│                                                                      │
│  ⚠️ CRITIQUE: Sans cette étape, H-Optimus ne sait pas "où regarder" │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ ÉTAPE 1.5: CellPose Slave (cyto3) — CONDITIONNEL                    │
├─────────────────────────────────────────────────────────────────────┤
│  Trigger:  Si organe requiert N/C ratio (Thyroid, Bladder)          │
│  Action:   CellPose cyto3 segmente cytoplasme                       │
│  Sortie:   N masques cytoplasme (matching avec noyaux)              │
│                                                                      │
│  Skip:     Cervix (SIPaKMeD) ne nécessite PAS N/C ratio            │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ ÉTAPE 2: Génération de Patches (Le Crop)                            │
├─────────────────────────────────────────────────────────────────────┤
│  Action:  Pour chaque bounding box:                                 │
│           1. Crop la région autour du noyau                          │
│           2. Padding blanc (PadIfNeeded) → 224×224                   │
│           3. Associer masque correspondant                           │
│                                                                      │
│  Sortie:  N tuples (image_224x224, masque_nuclei, masque_cyto)      │
│           Exemple: 50 patches prêts pour analyse                     │
└─────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PHASE PARALLÈLE (Extraction Features — Par Patch)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pour CHAQUE patch (itération sur les 50 cellules):

┌──────────────────────────────────┬──────────────────────────────────┐
│ ÉTAPE 3A: Branche Visuelle       │ ÉTAPE 3B: Branche Mathématique   │
│ (Le Cerveau — H-Optimus)         │ (Le Calculateur — Morphométrie)  │
├──────────────────────────────────┼──────────────────────────────────┤
│  Input:  Image RGB 224×224       │  Input:  Masques (nuclei + cyto) │
│                                  │                                  │
│  Action: H-Optimus-0 encode      │  Action: Calcul 14 features:     │
│          • CLS token extraction  │          1. Area                 │
│          • ViT-Giant/14          │          2. Perimeter            │
│          • Pré-entraîné gelé     │          3. Eccentricity         │
│                                  │          4. Solidity             │
│  Sortie: Embedding 1536D         │          5. N/C Ratio            │
│          (texture, couleur, ADN) │          6-14. Haralick features │
│                                  │          + Canal H (Ruifrok)     │
│                                  │                                  │
│                                  │  Sortie: Vecteur 14D             │
│                                  │          (mesures géométriques)  │
└──────────────────────────────────┴──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ ÉTAPE 4: Fusion (Concatenation Multi-Modale)                        │
├─────────────────────────────────────────────────────────────────────┤
│  Action:  Coller les deux vecteurs ensemble (opération "Frankenstein")│
│                                                                      │
│  Vecteur_Final = [Embedding_Optimus (1536)] + [Features_Morpho (14)]│
│                = Vecteur 1550D                                       │
│                                                                      │
│  Ce vecteur contient:                                                │
│  • Vision profonde de la texture (H-Optimus)                         │
│  • Mesures objectives géométriques (Morphométrie)                    │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ ÉTAPE 5: Décision (Classification Head)                             │
├─────────────────────────────────────────────────────────────────────┤
│  Input:   Vecteur 1550D                                              │
│                                                                      │
│  Action:  MLP léger (3 couches):                                     │
│           • Linear(1550, 512) + ReLU + Dropout(0.3)                 │
│           • Linear(512, 128) + ReLU + Dropout(0.2)                  │
│           • Linear(128, num_classes)                                 │
│           • Softmax                                                  │
│                                                                      │
│  Sortie:  Classe finale (ex: "Carcinoma in situ")                   │
│           + Score confiance (ex: 0.92)                               │
└─────────────────────────────────────────────────────────────────────┘

OUTPUT: Rapport pour les 50 cellules
        "5 cellules suspectes détectées sur 50 analysées"
```

---

## 🔍 Pourquoi Pas "Parallèle Pur" ?

### Scénario Impossible (Si Parallèle Pur)

```
❌ ARCHITECTURE NAÏVE (IMPOSSIBLE):

Image 1024×1024
    ├─→ CellPose → Détecte 50 cellules
    └─→ H-Optimus → ??? Comment analyser l'image entière ???
                        • H-Optimus attend 224×224
                        • Il ne fait pas de détection d'objets
                        • Il ne sait pas "où sont les cellules"
```

**Problème:** H-Optimus n'est PAS un détecteur comme YOLO ou Faster R-CNN.
**Solution:** CellPose DOIT venir en premier pour fournir les bounding boxes.

---

## 📊 Comparaison V13 vs V14

| Aspect | V13 Histologie (PanNuke) | V14 Cytologie (SIPaKMeD) |
|--------|--------------------------|--------------------------|
| **Tâche** | Segmentation instance | Détection + Classification |
| **Input** | Patch 224×224 pré-découpé | WSI 1024×1024 (N cellules) |
| **H-Optimus Role** | Backbone + Décodeur FPN | Feature extractor gelé |
| **H-Optimus Output** | Maps (NP, HV, NT) 224×224 | Embedding 1536D (CLS token) |
| **Post-processing** | Watershed (HV-guided) | MLP 3 couches |
| **Architecture** | Bout-à-bout trainable | Feature fusion |
| **Détection Cellules** | Fait par Watershed | Fait par CellPose Master |
| **Complexité** | Élevée (FPN Chimique) | Simple (MLP léger) |
| **Latence** | ~2s par patch | ~0.5s par cellule |

---

## 🎯 Rôles des Composants

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DIVISION DU TRAVAIL                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CellPose Master (nuclei)                                            │
│  ├─ Rôle: LOCALISATION (Où sont les cellules?)                      │
│  ├─ Output: Bounding boxes + Masques noyaux                         │
│  └─ Analogie: Le "Détective" qui trouve les suspects                │
│                                                                      │
│  CellPose Slave (cyto3)                                              │
│  ├─ Rôle: CONTEXTE (Quelle est la taille du cytoplasme?)            │
│  ├─ Output: Masques cytoplasme (si requis)                          │
│  └─ Analogie: Le "Mesureur" qui calcule les proportions             │
│                                                                      │
│  H-Optimus-0                                                         │
│  ├─ Rôle: ENCODAGE (Quelle est l'essence visuelle de la cellule?)   │
│  ├─ Output: Embedding 1536D (texture, couleur, motifs)              │
│  └─ Analogie: Le "Photographe Expert" qui capture la texture        │
│                                                                      │
│  Morphométrie                                                        │
│  ├─ Rôle: MESURE (Quelles sont les dimensions objectives?)          │
│  ├─ Output: 14 features géométriques + Canal H                      │
│  └─ Analogie: Le "Géomètre" qui mesure forme et taille              │
│                                                                      │
│  MLP Classification Head                                             │
│  ├─ Rôle: DÉCISION (Quel est le diagnostic final?)                  │
│  ├─ Output: Classe (7 classes SIPaKMeD) + Score confiance           │
│  └─ Analogie: Le "Juge" qui rend le verdict                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Principe Clé:** Chacun fait **UNE SEULE CHOSE**, mais la fait parfaitement.

- CellPose ne fait pas de classification → il localise
- H-Optimus ne fait pas de détection → il encode
- Morphométrie ne fait pas de décision → elle mesure
- MLP ne fait pas de segmentation → il classe

---

## 🚀 Optimisations Parallèles (Niveau Implémentation)

### Ce Qui Peut Être Parallélisé

```python
# Une fois qu'on a les N patches (après CellPose + Crop)

# ✅ PARALLÉLISATION POSSIBLE: Batch processing
images_batch = torch.stack([patch1, patch2, ..., patch50])  # (50, 3, 224, 224)

# Les 50 patches passent ensemble dans H-Optimus (batch inference)
embeddings = h_optimus(images_batch)  # (50, 1536) — GPU parallèle

# Les 50 masques peuvent aussi être traités en parallèle (CPU multi-thread)
with concurrent.futures.ThreadPoolExecutor() as executor:
    morpho_features = list(executor.map(compute_morphometry, masks))
```

**Gain:** Au lieu de 50 × 0.02s = 1s, on fait 50 patches en 0.1s (batch GPU).

### Ce Qui Ne Peut PAS Être Parallélisé

```python
# ❌ IMPOSSIBLE: H-Optimus avant CellPose
# On ne sait pas quoi encoder sans bounding boxes!

# ✅ CORRECT: Séquence obligatoire
bboxes, masks = cellpose.detect(image_1024)  # Étape 1
patches = crop_and_pad(image_1024, bboxes)    # Étape 2
embeddings = h_optimus(patches)               # Étape 3 (peut être batch)
```

---

## 📝 Correction Documentation

### Avant (Confus):

> "Architecture en Y: Router dirige vers Histo (V13) OU Cyto (Maître/Esclave + H-Optimus en parallèle)"

**Problème:** Suggère que CellPose et H-Optimus tournent simultanément.

### Après (Clair):

> "Pipeline V14 Cytologie:
> 1. **Phase Détection (Séquentiel):** CellPose Master → bounding boxes
> 2. **Phase Crop:** Génération patches 224×224 (padding blanc)
> 3. **Phase Features (Parallèle par patch):** H-Optimus + Morphométrie
> 4. **Phase Fusion:** Concaténation vecteurs
> 5. **Phase Classification:** MLP → verdict final"

---

## 🎓 Analogie Simple

**Imaginez une chaîne de diagnostic médical:**

1. **Le Radiologue (CellPose):**
   Regarde la radio complète → Encercle 10 zones suspectes
   *"Voici les 10 nodules à analyser"*

2. **Le Technicien (Crop + Padding):**
   Découpe les 10 zones encerclées → Prépare les échantillons

3. **L'Anatomopathologiste (H-Optimus) + Le Géomètre (Morpho):**
   Travaillent **en parallèle** sur chaque échantillon:
   - L'anatomo regarde la texture au microscope → Notes détaillées
   - Le géomètre mesure les dimensions → Tableau de chiffres

4. **Le Comité Médical (MLP):**
   Fusionne les notes + mesures → Verdict final

**Vous ne pouvez PAS demander à l'anatomo de travailler avant que le radiologue n'ait encerclé les zones!**

---

## ✅ Architecture Validée

L'expert a **100% raison**:

- **Séquentiel d'abord:** CellPose DOIT venir en premier (localisation)
- **Parallèle ensuite:** H-Optimus + Morpho tournent ensemble (sur chaque patch)
- **Fusion finale:** Concaténation + MLP

**Cette architecture est:**
- ✅ Logiquement cohérente
- ✅ Techniquement réalisable
- ✅ Optimisée pour la production (batch inference)

---

## 🔄 Impact sur le Code

### Structure Recommandée

```python
class CytologyPipeline:
    def __init__(self):
        self.cellpose_master = CellPoseNuclei()
        self.cellpose_slave = CellPoseCyto3()  # Conditionnel
        self.h_optimus = HOptimus0(frozen=True)
        self.morphometry = MorphometryEngine()
        self.classifier = MLPClassificationHead(input_dim=1550, num_classes=7)

    def predict(self, wsi_tile):
        """
        Pipeline complet V14 Cytologie

        Args:
            wsi_tile: Image (H, W, 3), ex: 1024×1024

        Returns:
            List[CellPrediction]: Résultats pour chaque cellule détectée
        """
        # ════════════════════════════════════════════════════════════
        # PHASE 1: DÉTECTION (Séquentiel)
        # ════════════════════════════════════════════════════════════
        bboxes, nuclei_masks = self.cellpose_master.detect(wsi_tile)

        if self.organ_requires_nc_ratio:
            cyto_masks = self.cellpose_slave.detect(wsi_tile)
        else:
            cyto_masks = [None] * len(bboxes)

        # ════════════════════════════════════════════════════════════
        # PHASE 2: CROP + PADDING
        # ════════════════════════════════════════════════════════════
        patches = []
        for bbox in bboxes:
            patch = crop_region(wsi_tile, bbox)
            patch = pad_to_224(patch, value=255)  # Padding blanc
            patches.append(patch)

        patches_tensor = torch.stack(patches)  # (N, 3, 224, 224)

        # ════════════════════════════════════════════════════════════
        # PHASE 3: EXTRACTION FEATURES (Parallèle — Batch)
        # ════════════════════════════════════════════════════════════

        # Branche A: H-Optimus (GPU batch)
        with torch.no_grad():
            embeddings = self.h_optimus(patches_tensor)  # (N, 1536)

        # Branche B: Morphométrie (CPU multi-thread)
        morpho_features = self.morphometry.batch_compute(
            nuclei_masks, cyto_masks
        )  # (N, 14)

        # ════════════════════════════════════════════════════════════
        # PHASE 4: FUSION
        # ════════════════════════════════════════════════════════════
        fused_features = torch.cat([embeddings, morpho_features], dim=1)  # (N, 1550)

        # ════════════════════════════════════════════════════════════
        # PHASE 5: CLASSIFICATION
        # ════════════════════════════════════════════════════════════
        logits = self.classifier(fused_features)  # (N, 7)
        probs = torch.softmax(logits, dim=1)
        classes = torch.argmax(probs, dim=1)

        # Construire résultats
        results = []
        for i in range(len(bboxes)):
            results.append(CellPrediction(
                bbox=bboxes[i],
                class_name=self.class_names[classes[i]],
                confidence=probs[i, classes[i]].item(),
                embedding=embeddings[i].cpu().numpy(),
                morpho_features=morpho_features[i]
            ))

        return results
```

---

## 📚 Références Mises à Jour

- **CellPose:** Modèle de détection (généraliste, pré-entraîné)
- **H-Optimus-0:** Feature extractor ViT-Giant/14 (1.1B params, gelé)
- **Morphométrie:** Calculs géométriques (OpenCV + scikit-image)
- **Classification Head:** MLP PyTorch (1550 → 512 → 128 → 7)

---

**Auteur:** Claude Code
**Validation Experte:** 2026-01-19
**Status:** ✅ Architecture Validée
