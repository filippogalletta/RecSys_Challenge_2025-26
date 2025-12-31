import numpy as np
import json
from typing import Dict, Tuple


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def generate_base_features(n: int, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Generate base numeric features x1, x2, x3 without missing values.
    We create some correlation between them so MAR has structure to exploit.
    """
    rng = np.random.default_rng(seed)

    # x1 ~ N(0, 1)
    x1 = rng.normal(loc=0.0, scale=1.0, size=n)

    # x2 = 0.5 * x1 + noise
    x2 = 0.5 * x1 + rng.normal(loc=0.0, scale=0.7, size=n)

    # x3 = -0.3 * x1 + 0.8 * x2 + noise
    x3 = -0.3 * x1 + 0.8 * x2 + rng.normal(loc=0.0, scale=0.5, size=n)

    return {"x1": x1, "x2": x2, "x3": x3}


def generate_target_columns(features: Dict[str, np.ndarray],
                            n: int,
                            seed: int = 123) -> Dict[str, np.ndarray]:
    """
    Generate fully observed target columns BEFORE introducing missingness.
    MAR targets remain correlated with base features.
    MNAR targets are also correlated with base features (so MCAR tests
    can detect dependence), but their missingness pattern will later be
    generated via a nonlinear function of their own value.
    """
    rng = np.random.default_rng(seed)
    x1, x2, x3 = features["x1"], features["x2"], features["x3"]

    # ------------------------------------------------------------
    # Base continuous targets correlated with features
    # ------------------------------------------------------------
    y_base1 = 1.5 * x1 + 0.5 * x2 + rng.normal(0, 1, n)
    y_base2 = -x2 + 0.8 * x3 + rng.normal(0, 1, n)

    # ------------------------------------------------------------
    # Targets for MCAR and MAR keep correlation with x1,x2,x3
    # ------------------------------------------------------------
    y_mcar1 = y_base1
    y_mcar2 = y_base2

    y_mar1 = 0.8 * y_base1 + 0.2 * x3 + rng.normal(0, 1, n)
    y_mar2 = -0.5 * y_base2 + 0.3 * x1 + rng.normal(0, 1, n)

    # ------------------------------------------------------------
    # MNAR targets: STILL correlated with base features,
    # but missingness (in apply_mnar) will be nonlinear in y.
    # ------------------------------------------------------------
    y_mnar1 = 1.2 * y_base1 + rng.normal(0, 1, n)
    y_mnar2 = -0.7 * y_base2 + rng.normal(0, 1, n)

    targets = {
        "y_mcar1": y_mcar1,
        "y_mcar2": y_mcar2,
        "y_mar1": y_mar1,
        "y_mar2": y_mar2,
        "y_mnar1": y_mnar1,
        "y_mnar2": y_mnar2,
    }

    return targets


def apply_mcar(mask_col: np.ndarray,
               p_missing: float,
               seed: int) -> np.ndarray:
    """
    Introduce MCAR missingness: each entry independently becomes NaN
    with probability p_missing.
    """
    rng = np.random.default_rng(seed)
    n = mask_col.shape[0]
    missing = rng.random(n) < p_missing
    col_with_missing = mask_col.copy()
    col_with_missing[missing] = np.nan
    return col_with_missing


def apply_mar(mask_col: np.ndarray,
              x1: np.ndarray,
              x2: np.ndarray,
              seed: int,
              base_prob: float = 0.15,
              strength: float = 1.2) -> np.ndarray:
    """
    Introduce MAR missingness: P(missing) depends on observed x1, x2.
    We build a logistic model and then sample Bernoulli.
    """
    rng = np.random.default_rng(seed)
    # Standardize predictors to keep logits in a reasonable range
    x1_std = (x1 - x1.mean()) / (x1.std() + 1e-8)
    x2_std = (x2 - x2.mean()) / (x2.std() + 1e-8)

    # Linear combination for logit
    # base_prob controls baseline missing rate; strength controls dependence
    logit = np.log(base_prob / (1 - base_prob)) + strength * (0.7 * x1_std - 0.5 * x2_std)
    p_missing = sigmoid(logit)

    # Ensure probabilities are not extreme
    p_missing = np.clip(p_missing, 0.02, 0.8)

    missing = rng.random(mask_col.shape[0]) < p_missing
    col_with_missing = mask_col.copy()
    col_with_missing[missing] = np.nan
    return col_with_missing


def apply_mnar(mask_col: np.ndarray,
               seed: int,
               base_prob: float = 0.15,
               strength: float = 1.0) -> np.ndarray:
    """
    Introduce MNAR missingness: P(missing) depends on a nonlinear
    function of the variable itself. The variable is still correlated
    with other features, so MCAR is rejected, but the nonlinearity
    makes MAR detection harder.
    """
    rng = np.random.default_rng(seed)

    y = mask_col
    y_std = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-8)

    # Nonlinear transform: oscillatory + cubic
    nonlinear = np.sin(2.0 * y_std) + 0.3 * np.power(y_std, 3)

    logit0 = np.log(base_prob / (1 - base_prob))
    logit = logit0 + strength * nonlinear

    p_missing = 1.0 / (1.0 + np.exp(-logit))
    p_missing = np.clip(p_missing, 0.02, 0.8)

    missing = rng.random(mask_col.shape[0]) < p_missing
    col_with_missing = mask_col.copy()
    col_with_missing[missing] = np.nan
    return col_with_missing


def make_synthetic_dataset(n: int = 2000,
                           seed_features: int = 42,
                           seed_targets: int = 123) -> Tuple[np.ndarray, Dict[str, str], Dict[str, np.ndarray]]:
    """
    Generate a full synthetic dataset with:
      - base features x1, x2, x3 (no missing)
      - MCAR columns y_mcar1, y_mcar2
      - MAR columns y_mar1, y_mar2
      - MNAR columns y_mnar1, y_mnar2

    Returns:
      - X: numeric matrix (n x m) with NaNs as missing
      - true_mechanisms: dict col_name -> mechanism ("none", "MCAR", "MAR", "MNAR")
      - columns: dict col_name -> 1D numpy array (for convenience)
    """
    # 1) Base features
    features = generate_base_features(n=n, seed=seed_features)

    # 2) Targets (fully observed)
    targets_full = generate_target_columns(features, n=n, seed=seed_targets)

    # 3) Introduce missingness
    x1, x2, x3 = features["x1"], features["x2"], features["x3"]

    y_mcar1 = apply_mcar(targets_full["y_mcar1"], p_missing=0.25, seed=1)
    y_mcar2 = apply_mcar(targets_full["y_mcar2"], p_missing=0.15, seed=2)

    y_mar1 = apply_mar(targets_full["y_mar1"], x1=x1, x2=x2, seed=3,
                       base_prob=0.10, strength=1.5)
    y_mar2 = apply_mar(targets_full["y_mar2"], x1=x1, x2=x2, seed=4,
                       base_prob=0.20, strength=1.0)

    y_mnar1 = apply_mnar(targets_full["y_mnar1"], seed=5,
                         base_prob=0.15, strength=0.6)
    y_mnar2 = apply_mnar(targets_full["y_mnar2"], seed=6,
                         base_prob=0.10, strength=0.6)

    # 4) Assemble all columns in a fixed order
    columns = {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "y_mcar1": y_mcar1,
        "y_mcar2": y_mcar2,
        "y_mar1": y_mar1,
        "y_mar2": y_mar2,
        "y_mnar1": y_mnar1,
        "y_mnar2": y_mnar2,
    }

    col_names = list(columns.keys())
    X = np.column_stack([columns[name] for name in col_names])

    true_mechanisms = {
        "x1": "none",
        "x2": "none",
        "x3": "none",
        "y_mcar1": "MCAR",
        "y_mcar2": "MCAR",
        "y_mar1": "MAR",
        "y_mar2": "MAR",
        "y_mnar1": "MNAR",
        "y_mnar2": "MNAR",
    }

    return X, true_mechanisms, columns


if __name__ == "__main__":
    # Example usage: generate dataset and save to CSV + metadata JSON
    X, true_mech, cols = make_synthetic_dataset(n=2000)

    # Save matrix to CSV (for quick inspection or import into SystemDS)
    np.savetxt("synthetic_missingness.csv", X, delimiter=",", fmt="%.6f")

    # Save column names in order
    col_names = list(cols.keys())
    with open("synthetic_missingness_columns.json", "w") as f:
        json.dump({"columns": col_names}, f, indent=2)

    # Save ground-truth mechanisms
    with open("synthetic_missingness_mechanisms.json", "w") as f:
        json.dump(true_mech, f, indent=2)

    print("Dataset saved to synthetic_missingness.csv")
    print("Columns:", col_names)
    print("True mechanisms:", true_mech)