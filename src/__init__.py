"""
Gorgonzola Racing Team - RecSys Challenge 2025/26
Politecnico di Milano

Package contenente i moduli per la Candidate Generation (Stage 1),
Feature Engineering e Reranking con XGBoost (Stage 2).
"""

from .candidate_generation import TripleIntegratedHierarchicalHybridRecommender
from .features import feature_populator, optimize_df
from .reranker import XGBoostRerankerRecommender, generate_submission
from .cross_validation import (
    split_train_in_five_percentage_global_sample,
    generate_kfold_splits,
    build_fold_training_dataset,
    evaluate_kfold_ranker
)

__all__ = [
    "TripleIntegratedHierarchicalHybridRecommender",
    "feature_populator",
    "optimize_df",
    "XGBoostRerankerRecommender",
    "generate_submission",
    "split_train_in_five_percentage_global_sample",
    "generate_kfold_splits",
    "build_fold_training_dataset",
    "evaluate_kfold_ranker"
]

