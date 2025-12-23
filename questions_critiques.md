# Questions Critiques à Résoudre

## Q1: Qu'est-ce qui a été fait lors du Bug #4 ?

D'après CLAUDE.md :
- `scripts/evaluation/visualize_raw_predictions.py` créé
- `scripts/evaluation/test_watershed_params.py` créé  
- `results/DIAGNOSTIC_REPORT_LOW_RECALL.md` généré
- `image_00002_diagnosis.png` créé

**À FAIRE:** Regarder ces fichiers pour comprendre ce qui a été trouvé !

## Q2: Les données FIXED utilisent-elles les vraies instances ?

D'après CLAUDE.md, il y a :
- `prepare_family_data.py` (OLD - avec connectedComponents ?)
- `prepare_family_data_FIXED.py` (NEW - avec vraies instances ?)

**À VÉRIFIER:** Quel script a été utilisé pour le training actuel ?

## Q3: Comment HoVer-Net original atteint AJI 0.68 ?

**À RECHERCHER dans le paper Graham et al. 2019:**
- Comment ils gèrent les cellules adjacentes ?
- Quel post-processing exactement ?
- Quels seuils watershed ?

## Q4: Nos HV maps sont-elles vraiment bonnes ?

HV MSE = 0.048 pendant training
Mais est-ce calculé sur :
- a) Instances fusionnées (connectedComponents) → Bon MSE mais gradients faibles
- b) Vraies instances PanNuke → Bon MSE et gradients forts

**À VÉRIFIER:** Inspecter les HV maps brutes prédites par le modèle

## Q5: Le watershed actuel est-il standard ?

**À COMPARER avec HoVer-Net original:**
- Même algorithme ?
- Mêmes seuils ?
- Mêmes paramètres ?
