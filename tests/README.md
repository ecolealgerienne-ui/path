# Tests - CellViT-Optimus

Suite de tests pour garantir la qualité et la cohérence du code.

## Structure

```
tests/
├── unit/                          # Tests unitaires rapides
│   ├── test_preprocessing.py      # Tests preprocessing (constants, functions)
│   └── test_model_loading.py      # Tests chargement modèles (à créer)
│
├── integration/                   # Tests d'intégration
│   ├── test_preprocessing_consistency.py  # Cohérence train/inference
│   └── test_pipeline_e2e.py       # Pipeline complet (à créer)
│
└── fixtures/                      # Données de test
    └── sample_images/             # Images de test (à ajouter)
```

## Installation

```bash
# Installer pytest
pip install pytest pytest-cov

# Optionnel: markers
pip install pytest-xdist  # Exécution parallèle
```

## Exécution

### Tests Rapides (Recommandé)

Exécute TOUS les tests SAUF les tests lents (qui nécessitent CUDA + téléchargement modèles):

```bash
pytest tests/ -v -m "not slow"
```

**Durée:** ~5-10 secondes

### Tests Complets (CI/CD)

Exécute TOUS les tests, y compris les tests lents:

```bash
pytest tests/ -v
```

**Durée:** ~2-3 minutes (première fois, télécharge H-optimus-0)

**Requis:**
- CUDA disponible
- Token HuggingFace configuré
- Connexion internet

### Tests Spécifiques

```bash
# Seulement preprocessing
pytest tests/unit/test_preprocessing.py -v

# Seulement cohérence
pytest tests/integration/test_preprocessing_consistency.py -v -m "not slow"

# Avec coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Tests Parallèles

```bash
# Exécution parallèle (plus rapide)
pytest tests/ -v -n auto
```

## Markers

Les tests utilisent des markers pour catégoriser:

| Marker | Description | Usage |
|--------|-------------|-------|
| `slow` | Tests lents (CUDA + modèles) | `-m "not slow"` pour skip |
| `unit` | Tests unitaires | `-m unit` |
| `integration` | Tests d'intégration | `-m integration` |
| `requires_data` | Nécessite données PanNuke | `-m "not requires_data"` pour skip |

**Exemples:**

```bash
# Seulement tests rapides
pytest -m "not slow"

# Seulement tests unitaires
pytest -m unit

# Seulement tests d'intégration, sans les lents
pytest -m "integration and not slow"
```

## Tests Critiques (Non-Régression)

Ces tests garantissent qu'on ne répète pas les bugs historiques:

### 1. test_uint8_vs_float64_equivalence

**Bug:** 2025-12-20 - ToPILImage multiplie floats par 255

**Garantit:** Conversion uint8 correcte avant ToPILImage

```bash
pytest tests/integration/test_preprocessing_consistency.py::TestPreprocessingDeterminism::test_uint8_vs_float64_equivalence -v
```

**Durée:** ~1 seconde

### 2. test_cls_std_in_expected_range

**Bug:** 2025-12-21 - blocks[23] sans LayerNorm → CLS std ~0.28

**Garantit:** forward_features() utilisé (CLS std ~0.77)

```bash
pytest tests/integration/test_preprocessing_consistency.py::TestCLSStdRange::test_cls_std_in_expected_range -v
```

**Durée:** ~30 secondes (télécharge modèle)

**Requis:** CUDA + Token HF

### 3. test_no_local_hoptimus_mean

**Garantit:** Aucune duplication de constantes HOPTIMUS_MEAN

```bash
pytest tests/integration/test_preprocessing_consistency.py::TestConstantsNotDuplicated::test_no_local_hoptimus_mean -v
```

**Durée:** ~1 seconde

## Ajout de Nouvelles Images de Test

Pour améliorer la couverture, ajoutez des images dans `tests/fixtures/sample_images/`:

```bash
# Structure recommandée
tests/fixtures/sample_images/
├── breast_01.png       # Cancer du sein
├── colon_01.png        # Côlon
├── prostate_01.png     # Prostate
└── ...

# Les tests les chargeront automatiquement
pytest tests/integration/test_preprocessing_consistency.py::TestEndToEndPipeline::test_real_image_if_available -v
```

## CI/CD

### GitHub Actions (Recommandé)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run fast tests
      run: pytest tests/ -v -m "not slow" --cov=src

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Tests Lents (CUDA)

Pour les tests nécessitant CUDA, utiliser un runner avec GPU:

```yaml
  test-cuda:
    runs-on: [self-hosted, gpu]

    steps:
    - name: Run all tests
      run: pytest tests/ -v
      env:
        HF_TOKEN: ${{ secrets.HUGGINGFACE_TOKEN }}
```

## Debugging

### Verbose Output

```bash
# Afficher print() statements
pytest tests/ -v -s

# Afficher traceback complet
pytest tests/ -v --tb=long

# Arrêter au premier échec
pytest tests/ -v -x
```

### Tests Individuels

```bash
# Exécuter UN seul test
pytest tests/unit/test_preprocessing.py::TestConstants::test_constants_are_tuples -v

# Pattern matching
pytest tests/ -k "uint8" -v
```

### PDB Debugging

```bash
# Lancer PDB à l'échec
pytest tests/ -v --pdb

# Lancer PDB immédiatement
pytest tests/ -v --trace
```

## Statistiques

```bash
# Durée de chaque test
pytest tests/ -v --durations=10

# Coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Bonnes Pratiques

1. **Toujours exécuter les tests rapides avant commit:**
   ```bash
   pytest tests/ -m "not slow" -v
   ```

2. **Ajouter tests pour chaque bug fixé** (non-régression)

3. **Garder les tests rapides (<1s chacun)** sauf si marker `slow`

4. **Ne pas skipper les tests critiques** (uint8, cls_std, duplications)

5. **Utiliser fixtures pour données réutilisables**

---

**Dernière mise à jour:** 2025-12-22
**Auteur:** Claude Code
