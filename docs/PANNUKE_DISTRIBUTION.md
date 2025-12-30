# Distribution PanNuke par Organe et Famille

## Vue d'ensemble

Le dataset PanNuke contient **7,900 images** réparties sur **19 organes** et **5 familles**.

---

## Distribution par Organe (triée par nombre de samples)

| Rang | Organe | Samples | % Total | Famille |
|------|--------|---------|---------|---------|
| 1 | **Breast** | **2,437** | **31.6%** | Glandular |
| 2 | Colon | 1,323 | 17.2% | Digestive |
| 3 | Adrenal_gland | 487 | 6.3% | Glandular |
| 4 | Esophagus | 427 | 5.5% | Digestive |
| 5 | HeadNeck | 396 | 5.1% | Epidermal |
| 6 | Bile-duct | 379 | 4.9% | Digestive |
| 7 | Cervix | 325 | 4.2% | Urologic |
| 8 | Uterus | 216 | 2.8% | Urologic |
| 9 | Pancreatic | 213 | 2.8% | Glandular |
| 10 | Prostate | 207 | 2.7% | Glandular |
| 11 | Testis | 193 | 2.5% | Urologic |
| 12 | Thyroid | 191 | 2.5% | Glandular |
| 13 | Liver | 186 | 2.4% | Respiratory |
| 14 | Lung | 178 | 2.3% | Respiratory |
| 15 | Skin | 178 | 2.3% | Epidermal |
| 16 | Bladder | 149 | 1.9% | Urologic |
| 17 | Stomach | 145 | 1.9% | Digestive |
| 18 | Kidney | 141 | 1.8% | Urologic |
| 19 | Ovarian | 129 | 1.7% | Urologic |
| | **Total** | **7,900** | **100%** | |

---

## Distribution par Famille

| Famille | Organes | Samples | % Total |
|---------|---------|---------|---------|
| **Glandular** | Breast, Prostate, Thyroid, Pancreatic, Adrenal_gland | **3,535** | **44.7%** |
| **Digestive** | Colon, Stomach, Esophagus, Bile-duct | **2,274** | **28.8%** |
| **Urologic** | Kidney, Bladder, Testis, Ovarian, Uterus, Cervix | **1,153** | **14.6%** |
| **Epidermal** | Skin, HeadNeck | **574** | **7.3%** |
| **Respiratory** | Lung, Liver | **364** | **4.6%** |

---

## Observations Clés

1. **Déséquilibre majeur:** Breast seul représente 31.6% du dataset
2. **Glandular domine:** 44.7% des données (grâce à Breast)
3. **Respiratory minoritaire:** Seulement 4.6% des données, pourtant AJI 0.6872 ✅
4. **Top 3 organes:** Breast + Colon + Adrenal_gland = 55.1% du dataset

---

## Implications pour l'Entraînement

| Famille | Samples | Attendu | Raison |
|---------|---------|---------|--------|
| Glandular | 3,535 | AJI élevé | Plus grand dataset + homogénéité tissulaire |
| Digestive | 2,274 | AJI moyen | Bon volume mais tissus tubulaires variés |
| Urologic | 1,153 | AJI moyen | Dataset modéré, 6 organes différents |
| Epidermal | 574 | AJI plus bas | Petit dataset + tissus stratifiés complexes |
| Respiratory | 364 | Variable | Très petit mais structures ouvertes |

---

## Références

- [PanNuke Dataset Extension Paper](https://arxiv.org/pdf/2003.10778)
- [PanNuke Official Site](https://jgamper.github.io/PanNukeDataset/)
- Dataset: Gamper et al., ECDP 2019 + Extension 2020
