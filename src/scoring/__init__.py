"""
Malignancy Scoring Module - V14 Cytology

Implements the 6 Universal Criteria for nuclear atypia scoring,
validated by ISBI 2014 MITOS-ATYPIA Challenge.

Key exports:
- MalignancyScoringEngine: Main scoring engine
- MalignancyScore: Score dataclass
"""

from .malignancy_scoring import (
    MalignancyScoringEngine,
    MalignancyScore,
    validate_against_isbi_2014
)

__all__ = [
    'MalignancyScoringEngine',
    'MalignancyScore',
    'validate_against_isbi_2014'
]
