"""
Gorgonzola Racing Team - RecSys Challenge 2025/26
Politecnico di Milano

Package contenente i moduli per la Candidate Generation (Stage 1),
Feature Engineering e Reranking con XGBoost (Stage 2).
"""

from .candidate_generation import TripleIntegratedHierarchicalHybridRecommender
from .features import feature_populator, optimize_df
from .reranker import XGBoostRerankerRecommender, generate_submission

__all__ = [
    "TripleIntegratedHierarchicalHybridRecommender",
    "feature_populator",
    "optimize_df",
    "XGBoostRerankerRecommender",
    "generate_submission"
]
