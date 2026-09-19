"""
Custom Recommender Models package
"""

from .scaled_puresvd import ScaledPureSVDRecommender
from .implicit_als import FeatureCombinedImplicitALSRecommender

__all__ = [
    "ScaledPureSVDRecommender",
    "FeatureCombinedImplicitALSRecommender"
]
