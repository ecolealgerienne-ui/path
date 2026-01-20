# V14 Cytology — Documentation

> **Version:** 14.0 (Production Ready)
> **Date:** 2026-01-19
> **Statut:** ✅ Architecture Validée (Expert)

---

## 📋 Vue d'Ensemble

Ce dossier contient **toute la documentation** du système V14 Cytologie (Dubai Edition).

**Architecture Validée:**
> *"V14 = CellPose localise + Optimus comprend + Morphométrie quantifie + MLP décide"*

---

## 📚 Documents Principaux

### 1. [V14_CYTOLOGY_BRANCH.md](./V14_CYTOLOGY_BRANCH.md) — Spécifications Complètes

**Statut:** ✅ Master Document

**Contenu:**
- Vue d'ensemble architecture en "Y"
- Specs techniques validées expert (2026-01-19)
- Pipeline 5 étapes (Séquentiel → Parallèle → Fusionné)
- 20 features morphométriques (table complète)
- Architecture MLP avec BatchNorm
- Macenko router-dependent
- Métriques Safety First
- Matrice de décision par organe

**Quand consulter:** Point d'entrée principal pour comprendre V14 Cytologie

---

### 2. [V14_PIPELINE_EXECUTION_ORDER.md](./V14_PIPELINE_EXECUTION_ORDER.md) — Ordre d'Exécution

**Statut:** 🔥 CRITIQUE

**Contenu:**
- **Clarification architecturale majeure:** Séquentiel PUIS Parallèle (pas "parallèle pur")
- Explication pourquoi CellPose DOIT venir AVANT H-Optimus
- 5 étapes détaillées avec diagrammes
- Rôles des composants (détection, encodage, mesure, décision)
- Optimisations batch (GPU/CPU)
- Comparaison V13 vs V14

**Quand consulter:** Avant d'implémenter le pipeline (essentiel développeurs)

---

### 3. [V14_MASTER_SLAVE_ARCHITECTURE.md](./V14_MASTER_SLAVE_ARCHITECTURE.md) — CellPose Dual-Model

**Statut:** 🎯 Architecture Pivot

**Contenu:**
- Orchestration CellPose Master (nuclei) + Slave (cyto3)
- Matrice de décision par organe (5 profils)
- Gains mesurés: 2× performance, 46% économie GPU
- KPIs critiques (Sensibilité > 0.98)
- Business model (4 packages €5k-€12k)
- Gestion cas d'erreur (noyaux orphelins)

**Quand consulter:** Pour comprendre la phase Segmentation (Étapes 1-4 du pipeline)

---

### 4. [V14_MACENKO_STRATEGY.md](./V14_MACENKO_STRATEGY.md) — Normalisation Router-Dependent

**Statut:** ✅ Validé (Specs Expert + Résultats V13)

**Contenu:**
- **Principe:** Macenko ON pour Cytologie, OFF pour Histologie
- Analyse technique conflit Ruifrok/Macenko (V13 -4.3% AJI)
- Pourquoi Macenko OK en V14 (pas de FPN Chimique)
- Code production preprocessor adaptatif
- Tests non-régression V13
- Tests production Dubai (multi-scanners)

**Quand consulter:** Avant d'implémenter le preprocessing (Étape 2.5)

---

### 5. [V14_CYTOLOGY_STANDALONE_STRATEGY.md](./V14_CYTOLOGY_STANDALONE_STRATEGY.md) — Stratégie Standalone

**Statut:** ⚠️ Archivé (Remplacé par approche Router)

**Contenu:**
- Approche standalone initiale (V14 Cytologie sans Router)
- Décision pivot: Router ajouté pour intégration V13/V14

**Quand consulter:** Contexte historique uniquement (non recommandé pour implémentation)

---

## 🗂️ Organisation Code Source

**Modules Python:**
- `src/cytology/morphometry.py` — 20 features morphométriques
- `src/cytology/models/cytology_classifier.py` — MLP avec BatchNorm
- `src/cytology/__init__.py` — Exports module

**Scripts Pipeline:**
- `scripts/cytology/` — 5 scripts exécution (masks, embeddings, features, train, eval)

**Documentation associée:**
- `scripts/cytology/README.md` — Guide pipeline complet

---

## 🎯 Workflow Lecture Recommandé

### Pour Développeurs (Première Fois)

1. **[V14_CYTOLOGY_BRANCH.md](./V14_CYTOLOGY_BRANCH.md)** — Comprendre l'architecture globale
2. **[V14_PIPELINE_EXECUTION_ORDER.md](./V14_PIPELINE_EXECUTION_ORDER.md)** — Ordre d'exécution (CRITIQUE)
3. **[V14_MASTER_SLAVE_ARCHITECTURE.md](./V14_MASTER_SLAVE_ARCHITECTURE.md)** — Détails CellPose
4. **[V14_MACENKO_STRATEGY.md](./V14_MACENKO_STRATEGY.md)** — Preprocessing
5. `scripts/cytology/README.md` — Guide pratique

### Pour Review/Validation

1. **[V14_CYTOLOGY_BRANCH.md](./V14_CYTOLOGY_BRANCH.md)** — Specs complètes
2. **[V14_PIPELINE_EXECUTION_ORDER.md](./V14_PIPELINE_EXECUTION_ORDER.md)** — Ordre exécution
3. Code source (`src/cytology/`)

### Pour Production/Déploiement

1. **[V14_MACENKO_STRATEGY.md](./V14_MACENKO_STRATEGY.md)** — Tests non-régression V13
2. **[V14_MASTER_SLAVE_ARCHITECTURE.md](./V14_MASTER_SLAVE_ARCHITECTURE.md)** — KPIs critiques
3. `scripts/cytology/README.md` — Pipeline exécution

---

## 📊 Métriques Prioritaires (Safety First)

| Métrique | Seuil Cible | Priorité | Document Référence |
|----------|-------------|----------|-------------------|
| **Sensibilité Malin** | **> 0.98** | 🔴 CRITIQUE | V14_MASTER_SLAVE_ARCHITECTURE.md |
| **FROC (FP/WSI @ 98% sens)** | **< 2.0** | 🔴 CRITIQUE | V14_MASTER_SLAVE_ARCHITECTURE.md |
| **Cohen's Kappa** | **> 0.80** | 🔴 CRITIQUE | V14_MASTER_SLAVE_ARCHITECTURE.md |
| IoU Noyau | > 0.85 | 🟡 Important | V14_CYTOLOGY_BRANCH.md |
| AP50 (COCO) | > 0.90 | 🟡 Important | V14_CYTOLOGY_BRANCH.md |

**Principe:** Ne JAMAIS rater un cancer (Sensibilité > Précision)

---

## 🔗 Références Externes

**Code Source:**
- `src/cytology/` — Modules Python
- `scripts/cytology/` — Scripts pipeline

**Documentation Projet:**
- `CLAUDE.md` — Documentation projet principale (lien vers ce dossier)
- `docs/datasets/` — Datasets cytologie (SIPaKMeD, Herlev, ISBI 2014)

**Datasets:**
- SIPaKMeD: 4,049 images cervicales (7 classes)
- Herlev: 917 images cervicales
- ISBI 2014: ~1,200 images breast histology (validation uniquement)

---

## 📝 Historique Versions

### Version 14.0 — 2026-01-19 (Production Ready)

**Changements Majeurs:**
- ✅ Architecture validée expert
- ✅ 20 features morphométriques complètes
- ✅ MLP avec BatchNorm (fusion multimodale)
- ✅ Macenko router-dependent (Cyto ON / Histo OFF)
- ✅ Pipeline ordre exécution clarifié (Séquentiel PUIS Parallèle)
- ✅ Modules Python production-ready (`src/cytology/`)

**Décisions Architecturales:**
- CellPose Master/Slave orchestration
- H-Optimus-0 figé (1.1B params)
- Focal Loss (déséquilibre classes)
- SINGLE SOURCE OF TRUTH (features sur masques CellPose)

**Prochaines Étapes:**
- Phase 1: Implémenter 5 scripts pipeline
- Phase 2: Training sur SIPaKMeD (4,049 images)
- Phase 3: Validation Safety First (Sensibilité > 0.98)

---

## 🚀 Quick Start

```bash
# Consulter architecture globale
cat docs/cytology/V14_CYTOLOGY_BRANCH.md

# Comprendre ordre exécution
cat docs/cytology/V14_PIPELINE_EXECUTION_ORDER.md

# Guide pratique pipeline
cat scripts/cytology/README.md
```

---

**Auteur:** V14 Cytology Branch
**Validation:** Expert (2026-01-19)
**Statut:** ✅ Production Ready
