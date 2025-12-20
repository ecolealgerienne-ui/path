"""
Module Active Learning pour CellViT-Optimus.

Permet de collecter les corrections des pathologistes
pour améliorer le modèle de manière itérative.
"""

from .active_learning import FeedbackCollector, FeedbackEntry, FeedbackType

__all__ = ["FeedbackCollector", "FeedbackEntry", "FeedbackType"]
