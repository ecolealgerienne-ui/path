# Sources de Téléchargement des Datasets d'Évaluation

Ce document liste les sources officielles et alternatives pour télécharger les datasets d'évaluation.

## 🥇 PanNuke (Priorité 1)

| Attribut | Valeur |
|----------|--------|
| **Images** | 7,901 (256×256 RGB) |
| **Classes** | 5 + background |
| **Organes** | 19 types |
| **Taille** | ~1.5 GB (compressé) |
| **Licence** | CC BY-NC-SA 4.0 |

### Sources de téléchargement

1. **Site officiel Warwick (recommandé)**
   - URL: https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/
   - Folds séparés: fold_1.zip, fold_2.zip, fold_3.zip
   - Téléchargement direct, pas d'authentification

2. **Zenodo (miroir officiel)**
   - URL: https://zenodo.org/record/3939982
   - Archive complète avec documentation
   - DOI: 10.5281/zenodo.3939982

### Citation

```bibtex
@article{gamper2020pannuke,
  title={PanNuke Dataset Extension, Insights and Baselines},
  author={Gamper, Jevgenij and Koohbanani, Navid Alemi and others},
  journal={arXiv preprint arXiv:2003.10778},
  year={2020}
}
```

---

## 🥈 CoNSeP (Priorité 2)

| Attribut | Valeur |
|----------|--------|
| **Images** | 41 (1000×1000 RGB) |
| **Classes** | 4 types de noyaux |
| **Taille** | ~70 MB |
| **Licence** | Recherche uniquement |

### Sources de téléchargement

⚠️ **Note:** Le téléchargement automatique peut échouer. Utiliser le téléchargement manuel.

1. **Site officiel Warwick**
   - URL: https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/
   - Fichier: `consep_dataset.zip`
   - ⚠️ Peut rediriger vers une page d'authentification

2. **Alternative via formulaire de contact**
   - Email: tia@warwick.ac.uk
   - Sujet: "CoNSeP Dataset Request"
   - Mentionner l'usage: recherche académique

3. **Dépôt GitHub HoVer-Net**
   - URL: https://github.com/vqdang/hover_net
   - Vérifier les Releases pour d'éventuels liens

### Structure attendue après extraction

```
consep/
├── Train/
│   ├── Images/  (16 images .png)
│   └── Labels/  (16 .mat files)
└── Test/
    ├── Images/  (14 images .png)
    └── Labels/  (14 .mat files)
```

### Citation

```bibtex
@article{graham2019hover,
  title={Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images},
  author={Graham, Simon and others},
  journal={Medical image analysis},
  volume={58},
  pages={101563},
  year={2019}
}
```

---

## 🥉 MoNuSAC (Priorité 3)

| Attribut | Valeur |
|----------|--------|
| **Images** | 209 |
| **Classes** | 4 types immunitaires |
| **Taille** | ~500 MB |
| **Licence** | CC BY-NC-SA 4.0 |

### Sources de téléchargement

1. **Hugging Face (recommandé)**
   - URL: https://huggingface.co/datasets/RationAI/MoNuSAC
   - Téléchargement via `datasets` library
   - Authentification HF optionnelle

2. **Site officiel MoNuSAC**
   - URL: https://monusac-2020.grand-challenge.org/
   - Inscription requise
   - Téléchargement manuel après approbation

### Citation

```bibtex
@article{verma2020monusac,
  title={MoNuSAC2020: A Multi-Organ Nuclei Segmentation and Classification Challenge},
  author={Verma, Ruchika and others},
  journal={IEEE Transactions on Medical Imaging},
  year={2020}
}
```

---

## 📦 Lizard (Dataset additionnel)

| Attribut | Valeur |
|----------|--------|
| **Images** | 291 (colon) |
| **Noyaux** | 500,000+ annotés |
| **Taille** | ~2 GB |
| **Licence** | CC BY-NC-SA 4.0 |

### Sources de téléchargement

1. **Site officiel Warwick**
   - URL: https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/
   - Fichier: `lizard_images_and_labels.zip`

### Citation

```bibtex
@article{graham2021lizard,
  title={Lizard: A Large-scale Dataset for Colonic Nuclear Instance Segmentation and Classification},
  author={Graham, Simon and others},
  journal={ICCV Workshops},
  year={2021}
}
```

---

## 🛠️ Outils de Téléchargement

### Script Python automatique

```bash
# Afficher les datasets disponibles
python scripts/evaluation/download_evaluation_datasets.py --info

# Télécharger PanNuke (fonctionne bien)
python scripts/evaluation/download_evaluation_datasets.py --dataset pannuke --folds 2

# Télécharger CoNSeP (peut échouer - voir manuel)
python scripts/evaluation/download_evaluation_datasets.py --dataset consep
```

### Script shell manuel (CoNSeP)

```bash
bash scripts/evaluation/download_consep_manual.sh
```

### Téléchargement avec wget/curl

```bash
# PanNuke Fold 2
wget https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip

# CoNSeP (peut nécessiter authentification)
wget https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep_dataset.zip
```

---

## 📝 Notes Importantes

### Licences et Usage

- **PanNuke, MoNuSAC, Lizard**: CC BY-NC-SA 4.0 (usage non-commercial)
- **CoNSeP**: Usage recherche uniquement, contacter les auteurs pour usage commercial

### Problèmes de Téléchargement

1. **CoNSeP échoue automatiquement**
   - Cause: Redirection vers page HTML
   - Solution: Téléchargement manuel depuis le site Warwick

2. **Zenodo lent**
   - Utiliser un gestionnaire de téléchargement (wget avec resume)
   - Télécharger pendant heures creuses

3. **Hugging Face nécessite authentification**
   - Créer un compte (gratuit)
   - Générer un token d'accès
   - `huggingface-cli login`

### Vérification de l'Intégrité

Après téléchargement, vérifier la taille :

| Dataset | Fichier | Taille attendue |
|---------|---------|----------------|
| PanNuke Fold 1 | fold_1.zip | ~500 MB |
| PanNuke Fold 2 | fold_2.zip | ~500 MB |
| PanNuke Fold 3 | fold_3.zip | ~500 MB |
| CoNSeP | consep_dataset.zip | ~70 MB |
| Lizard | lizard_images_and_labels.zip | ~2 GB |

Si le fichier fait < 1 MB, c'est probablement une page HTML de redirection.

---

## 🆘 Support

En cas de problème de téléchargement :

1. **Vérifier les logs**
   ```bash
   python scripts/evaluation/download_evaluation_datasets.py --dataset consep 2>&1 | tee download.log
   ```

2. **Contacter les auteurs**
   - Warwick TIA Lab: tia@warwick.ac.uk
   - Inclure: nom du dataset, erreur rencontrée, usage prévu

3. **Consulter la documentation**
   - README: `scripts/evaluation/README.md`
   - Spec: `docs/PLAN_EVALUATION_GROUND_TRUTH.md`

---

**Dernière mise à jour:** 2025-12-21
