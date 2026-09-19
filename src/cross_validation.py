"""
Cross-Validation & Out-of-Fold Framework for Recommender Systems
===============================================================
This module provides robust, leak-free 5-Fold cross-validation utilities
specifically designed for two-stage recommender systems.

Features:
- `split_train_in_five_percentage_global_sample`: Disjoint interaction-level sampling on sparse CSR matrices.
- `generate_kfold_splits`: Generator yielding (URM_train_fold, URM_test_fold) pairs.
- `build_fold_training_dataset`: Constructs unbiased Stage 2 training data using Stage 1 models trained strictly on train folds.
- `evaluate_kfold_ranker`: Evaluates an XGBRanker across folds to compute cross-validated Recall@K / MAP@K.
"""

from typing import Dict, List, Tuple, Generator, Any, Optional, Callable
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp


def split_train_in_five_percentage_global_sample(
    URM_all: sp.csr_matrix,
    train_percentages: Optional[List[float]] = None,
    seed: int = 42
) -> List[sp.csr_matrix]:
    """
    Splits a sparse user-rating matrix (URM) into disjoint sub-matrices based on global interaction sampling.

    Parameters
    ----------
    URM_all : sp.csr_matrix
        The complete interaction matrix to partition.
    train_percentages : list of float, optional
        Partition ratios (must sum to 1.0). Defaults to 5 equal splits [0.2, 0.2, 0.2, 0.2, 0.2].
    seed : int, default=42
        Random seed for reproducible interaction shuffling.

    Returns
    -------
    list of sp.csr_matrix
        List of 5 disjoint sparse CSR matrices with identical shape to URM_all.
    """
    if train_percentages is None:
        train_percentages = [0.2, 0.2, 0.2, 0.2, 0.2]

    if len(train_percentages) != 5:
        raise ValueError("train_percentages must contain exactly 5 elements.")
    if not np.isclose(sum(train_percentages), 1.0):
        raise ValueError("train_percentages must sum to 1.0.")

    URM_all_coo = sp.coo_matrix(URM_all)
    num_users, num_items = URM_all.shape
    nnz = URM_all.nnz

    rng = np.random.default_rng(seed)
    indices = np.arange(nnz, dtype=np.int32)
    rng.shuffle(indices)

    split_sizes = [int(nnz * p) for p in train_percentages]
    split_sizes[-1] = nnz - sum(split_sizes[:-1])  # Ensure exact sum
    cumulative_sizes = np.cumsum(split_sizes)

    sparse_matrices: List[sp.csr_matrix] = []
    start_idx = 0
    for end_idx in cumulative_sizes:
        fold_idx = indices[start_idx:end_idx]
        fold_matrix = sp.csr_matrix(
            (URM_all_coo.data[fold_idx], (URM_all_coo.row[fold_idx], URM_all_coo.col[fold_idx])),
            shape=(num_users, num_items),
            dtype=URM_all.dtype
        )
        sparse_matrices.append(fold_matrix)
        start_idx = end_idx

    return sparse_matrices


def generate_kfold_splits(
    URM_all: sp.csr_matrix,
    n_splits: int = 5,
    seed: int = 42
) -> Generator[Tuple[int, sp.csr_matrix, sp.csr_matrix], None, None]:
    """
    Generator yielding (fold_index, URM_train_fold, URM_test_fold) for cross-validation.

    Parameters
    ----------
    URM_all : sp.csr_matrix
        Complete interaction matrix.
    n_splits : int, default=5
        Number of folds.
    seed : int, default=42
        Random seed for split reproducibility.

    Yields
    ------
    tuple of (int, sp.csr_matrix, sp.csr_matrix)
        (fold_idx, URM_train_fold, URM_test_fold)
    """
    percentages = [1.0 / n_splits] * n_splits
    partitions = split_train_in_five_percentage_global_sample(URM_all, train_percentages=percentages, seed=seed)

    for i in range(n_splits):
        URM_test_fold = partitions[i]
        train_parts = [partitions[j] for j in range(n_splits) if j != i]
        URM_train_fold = sum(train_parts)
        yield i, URM_train_fold, URM_test_fold


def build_fold_training_dataset(
    URM_train_fold: sp.csr_matrix,
    URM_test_fold: sp.csr_matrix,
    candidate_generator: Any,
    other_algorithms: Dict[str, Any],
    feature_populator_func: Callable,
    cutoff: int = 50,
    target_users: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Builds a training dataset for a single cross-validation fold.
    """
    training_df = feature_populator_func(
        URM_train=URM_train_fold,
        recommender=candidate_generator,
        other_algorithms=other_algorithms,
        cutoff=cutoff,
        users_to_recommend=target_users
    )

    URM_test_coo = sp.coo_matrix(URM_test_fold)
    correct_recs = pd.DataFrame({"UserID": URM_test_coo.row, "ItemID": URM_test_coo.col})

    training_df = pd.merge(
        training_df,
        correct_recs,
        on=["UserID", "ItemID"],
        how="left",
        indicator="Exist"
    )
    training_df["Label"] = (training_df["Exist"] == "both").astype(int)
    training_df.drop(columns=["Exist"], inplace=True)

    training_df.sort_values(by=["UserID", "ItemID"], inplace=True)
    groups = training_df.groupby("UserID", sort=False).size().values

    y_train = training_df["Label"]
    X_train = training_df.drop(columns=["Label"])

    X_train["UserID"] = X_train["UserID"].astype(np.int32)
    X_train["ItemID"] = X_train["ItemID"].astype(np.int32)

    return {
        "X": X_train,
        "y": y_train,
        "groups": groups,
        "full_df": training_df,
        "URM_train": URM_train_fold,
        "URM_test": URM_test_fold
    }


def evaluate_kfold_ranker(
    fold_datasets: List[Dict[str, Any]],
    fold_datasets_val: List[Dict[str, Any]],
    xgb_ranker_cls: Any,
    xgb_params: Dict[str, Any],
    reranker_cls: Any,
    evaluator_cls: Any,
    cutoff: int = 20
) -> Dict[str, Any]:
    """
    Fits and evaluates XGBRanker across all pre-computed folds.
    """
    recalls = []
    maps = []

    for i in range(len(fold_datasets)):
        model = xgb_ranker_cls(**xgb_params)
        model.fit(
            fold_datasets[i]["X"],
            fold_datasets[i]["y"],
            group=fold_datasets[i]["groups"],
            verbose=False
        )

        reranker = reranker_cls(
            fold_datasets_val[i]["URM_train"],
            model,
            fold_datasets_val[i]["full_df"]
        )

        evaluator = evaluator_cls(fold_datasets_val[i]["URM_test"], cutoff_list=[cutoff])
        result_df, _ = evaluator.evaluateRecommender(reranker)

        rec = result_df.loc[cutoff, "RECALL"] if "RECALL" in result_df.columns else result_df["RECALL"].values[0]
        mp = result_df.loc[cutoff, "MAP"] if "MAP" in result_df.columns else result_df["MAP"].values[0]

        recalls.append(float(rec))
        maps.append(float(mp))

        del model, reranker, evaluator
        gc.collect()

    return {
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "map_mean": float(np.mean(maps)),
        "map_std": float(np.std(maps)),
        "fold_recalls": recalls,
        "fold_maps": maps
    }
