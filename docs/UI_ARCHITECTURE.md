# CellViT-Optimus UI — Architecture Modulaire

> **Version:** 2.0 (Refactorisation Décembre 2025)
> **Objectif:** Logique partagée, affichage différencié

---

## Vue d'ensemble

L'interface utilisateur CellViT-Optimus est composée de deux applications Gradio:

| Interface | Port | Audience | Style |
|-----------|------|----------|-------|
| **R&D Cockpit** (`app.py`) | 7860 | Chercheurs, développeurs | Technique, métriques détaillées |
| **Pathologiste** (`app_pathologist.py`) | 7861 | Cliniciens | Simplifié, langage médical |

**Principe fondamental:** La logique métier est **unique** (module `core`), seul l'affichage diffère (module `formatters`).

---

## Architecture

```
src/ui/
├── app.py                      # Interface R&D Cockpit
├── app_pathologist.py          # Interface Pathologiste
├── inference_engine.py         # Moteur IA (CellVitEngine)
├── organ_config.py             # Configuration organes/modèles
├── visualizations.py           # Fonctions de visualisation
├── export.py                   # Export PDF/CSV
│
├── core/                       # LOGIQUE PARTAGÉE
│   ├── __init__.py
│   ├── engine_ops.py           # Opérations moteur
│   └── export_ops.py           # Opérations export
│
└── formatters/                 # AFFICHAGE DIFFÉRENCIÉ
    ├── __init__.py
    ├── format_rnd.py           # Formatage R&D (technique)
    └── format_clinical.py      # Formatage clinique (simplifié)
```

---

## Module Core (`src/ui/core/`)

Le module `core` contient toute la logique métier partagée. Les fonctions retournent des **données brutes** (dicts, dataclasses), sans formatage d'affichage.

### État Global

```python
from src.ui.core import state

# Attributs:
state.engine          # CellVitEngine | None
state.current_result  # AnalysisResult | None
state.is_loading      # bool
```

### Opérations Moteur (`engine_ops.py`)

| Fonction | Description | Retour |
|----------|-------------|--------|
| `load_engine_core(organ, device)` | Charge le moteur | `Dict[success, organ, model_type, device, error]` |
| `change_organ_core(organ)` | Change l'organe actif | `Dict[success, organ, model_type, watershed_params, error]` |
| `analyze_image_core(image, ...)` | Analyse complète | `AnalysisOutput` (dataclass) |
| `on_image_click_core(x, y)` | Info noyau au clic | `Dict[found, nucleus_id, cell_type, ...]` |

### Opérations Export (`export_ops.py`)

| Fonction | Description | Retour |
|----------|-------------|--------|
| `export_pdf_core()` | Génère rapport PDF | `str` (chemin fichier) |
| `export_nuclei_csv_core()` | Export CSV noyaux | `str` (chemin fichier) |
| `export_summary_csv_core()` | Export CSV résumé | `str` (chemin fichier) |
| `export_json_core()` | Export JSON | `str` (contenu JSON) |

### AnalysisOutput (dataclass)

```python
@dataclass
class AnalysisOutput:
    success: bool
    result: Optional[AnalysisResult] = None
    overlay: Optional[np.ndarray] = None
    contours: Optional[np.ndarray] = None
    chart: Optional[np.ndarray] = None
    debug: Optional[np.ndarray] = None
    anomaly_overlay: Optional[np.ndarray] = None
    phase3_overlay: Optional[np.ndarray] = None
    phase3_debug: Optional[np.ndarray] = None
    error: Optional[str] = None
```

---

## Module Formatters (`src/ui/formatters/`)

Le module `formatters` contient les fonctions de formatage d'affichage. Chaque interface importe son module de formatage.

### Format R&D (`format_rnd.py`)

Style **technique** avec métriques détaillées, debug visible.

```python
from src.ui.formatters import (
    format_metrics_rnd,
    format_alerts_rnd,
    format_nucleus_info_rnd,
    format_load_status_rnd,
    format_organ_change_rnd,
)
```

**Caractéristiques:**
- Toutes les métriques brutes affichées
- Ratio I/E, index mitotique détaillés
- Phase 3 avec entropie, voisins Voronoï
- Paramètres watershed visibles

### Format Clinical (`format_clinical.py`)

Style **simplifié** avec langage clinique.

```python
from src.ui.formatters import (
    format_metrics_clinical,
    format_alerts_clinical,
    format_nucleus_info_clinical,
    format_identification_clinical,
    format_load_status_clinical,
    format_organ_change_clinical,
    format_confidence_badge,
    interpret_density,
    interpret_pleomorphism,
    interpret_mitotic_index,
)
```

**Caractéristiques:**
- Métriques interprétées ("Faible", "Modéré", "Élevé")
- Pas de valeurs brutes techniques
- Badge de confiance IA visuel
- Alertes en langage médical

---

## Pattern d'utilisation

### Dans app.py (R&D)

```python
from src.ui.core import state, load_engine_core, analyze_image_core
from src.ui.formatters import format_metrics_rnd, format_alerts_rnd

def load_engine(organ, device):
    result = load_engine_core(organ, device)
    return format_load_status_rnd(result)

def analyze_image(image, np_threshold, min_size, beta, min_distance):
    output = analyze_image_core(image, np_threshold, min_size, beta, min_distance)

    if not output.success:
        return output.overlay, output.contours, output.error, "", ...

    metrics = format_metrics_rnd(output.result, organ, family, is_dedicated)
    alerts = format_alerts_rnd(output.result)

    return output.overlay, output.contours, metrics, alerts, ...
```

### Dans app_pathologist.py (Clinique)

```python
from src.ui.core import state, load_engine_core
from src.ui.formatters import format_metrics_clinical, format_alerts_clinical

def load_engine(organ, device):
    result = load_engine_core(organ, device)
    return format_load_status_clinical(result)

def analyze_image(image):
    # Utilise les params watershed automatiques
    params = state.engine.watershed_params
    result = state.engine.analyze(image, watershed_params=params, ...)

    metrics = format_metrics_clinical(result, organ, family, is_dedicated)
    alerts = format_alerts_clinical(result)

    return overlay, identification, metrics, alerts, ...
```

---

## Comparaison des affichages

### Exemple: Densité cellulaire

| Interface | Affichage |
|-----------|-----------|
| **R&D** | `- Densité: **2847** noyaux/mm²` |
| **Clinique** | `**Densité cellulaire:** Élevée (2847/mm²)` |

### Exemple: Index mitotique

| Interface | Affichage |
|-----------|-----------|
| **R&D** | `- Index mitotique: *non calculé* — Signal IA: **activité modérée** (5 candidats)` |
| **Clinique** | `**Index mitotique:** 5/10 HPF (Modéré)` |

### Exemple: Alertes

| Interface | Affichage |
|-----------|-----------|
| **R&D** | `- 🔴 **Pléomorphisme sévère** — anisocaryose marquée` |
| **Clinique** | `**Anisocaryose sévère** — forte variation taille/forme nucléaire` |

---

## Avantages de l'architecture

1. **Single Source of Truth**: La logique métier est unique dans `core/`
2. **Pas de duplication**: Les calculs ne sont pas dupliqués entre interfaces
3. **Maintenance simplifiée**: Un bug corrigé dans `core/` l'est pour les deux interfaces
4. **Extensibilité**: Ajouter une nouvelle interface = nouveau fichier formatter
5. **Tests**: Le module `core/` peut être testé indépendamment de l'UI

---

## Lancer les interfaces

```bash
# R&D Cockpit (port 7860)
python -m src.ui.app --preload --organ Lung

# Interface Pathologiste (port 7861)
python -m src.ui.app_pathologist --preload --organ Lung

# Ou utiliser le script de gestion
./scripts/ui_manager.sh start cockpit
./scripts/ui_manager.sh start pathologist
./scripts/ui_manager.sh status
```

---

## Références

- [docs/UI_COCKPIT.md](./UI_COCKPIT.md) — Documentation détaillée du R&D Cockpit
- [CLAUDE.md](../CLAUDE.md) — Contexte projet complet
