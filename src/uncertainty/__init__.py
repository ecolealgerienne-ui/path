#!/usr/bin/env python3
"""
Module d'incertitude et sécurité pour CellViT-Optimus.

Couche 3 de l'architecture:
- Incertitude aléatorique (entropie NP/HV)
- Incertitude épistémique (Conformal Prediction)
- Détection OOD (distance Mahalanobis)
- Calibration locale (Temperature Scaling)

Sortie: {Fiable | À revoir | Hors domaine}
"""

from .uncertainty_estimator import UncertaintyEstimator, UncertaintyResult

__all__ = ['UncertaintyEstimator', 'UncertaintyResult']
