# Experimental Journey & Optimization Narrative

This folder contains the final experimental artifacts and logs tracing how **Gorgonzola Racing Team** converged on the winning two-stage recommender pipeline (**Recall@20 = 0.52522**).

---

## The Road to 0.52522: How We Built the Pipeline

```
+-------------------+      +-------------------------+      +--------------------------+      +----------------------+
| 1. Single Models  | ---> | 2. Hierarchical Stage 1 | ---> | 3. Leak-Free 5-Fold CV   | ---> | 4. Final Stage 2 GBDT|
| Max Recall ~0.47  |      | Recall@90 Ceiling >85%  |      | Out-Of-Fold Partitioning |      | Recall@20 = 0.52522  |
+-------------------+      +-------------------------+      +--------------------------+      +----------------------+
```

### 1. The Bottleneck of Single Models
Early experiments showed the complementary strengths and limitations of individual paradigms:
* **SLIM ElasticNet**: Exceptional precision on dense users, but computationally heavy and struggles with extreme long-tail items.
* **RP3beta**: Excellent long-tail discovery via 3-hop random walks, but noisy on high-degree items.
* **EASE_R**: Strong closed-form item-item correlations, but tends to over-recommend popular clusters.
* **Implicit ALS**: Strong user latent representation, but lacks localized graph neighborhood precision.

No single algorithm could exceed ~0.47 Recall@20 on the competition benchmark.

### 2. Multi-Level Hierarchical Blending (M3)
To overcome these individual limits without exploding computational cost:
1. **Similarity Fusion ($W_{12}$)**: Convex combination of SLIM and EASE ($\alpha = 0.157$).
2. **Graph Blending ($W_{\text{final}}$)**: Blending $W_{12}$ with RP3beta ($\beta = 0.088$) creates a unified sparse similarity matrix.
3. **Latent Score Blending**: Combining CustomKNN scores with iALS ratings ($\gamma = 0.127$).

**Outcome**: A compact pool of **90 candidates per user** capturing **>85%** of all relevant interactions (Recall Ceiling).

### 3. Solving Target Leakage with 5-Fold Out-Of-Fold CV
When training an XGBoost reranker, computing features using models fit on the same interactions used for ground-truth labels creates severe **target leakage**, causing the trees to memorize training artifacts and fail on test data.

**Our Fix**:
* Partitioned the sparse URM globally into 5 disjoint subsets ($20\%$ each).
* In each fold $i$, base models and candidate generators were trained strictly on $URM_{\text{train}}^{(i)}$.
* 70+ tabular features were generated on unseen interactions before matching with $URM_{\text{test}}^{(i)}$.

### 4. Bayesian Optimization & Convergence (`xgb_optuna_trials.txt`)
We executed a 100-trial Bayesian Optimization study (TPE sampler) searching over 11 tree architecture and regularization parameters:
* **Top Configuration (Trial 75)** achieved **MAP@20 = 0.29576**:
  * `n_estimators`: 2576
  * `learning_rate`: 0.0048
  * `max_depth`: 7, `max_leaves`: 236 (`grow_policy: lossguide`)
  * `subsample`: 0.63, `colsample_bytree`: 0.57
  * `reg_alpha`: 0.0226, `reg_lambda`: 0.00009

### 5. Final Verification & Feature Importance (`feature_importance.png`)
Feature analysis revealed that top performance relies on:
1. Stage 1 composite hybrid scores (`SLIMElastic_Score`, `EASE_R_Score`, `IALS_Score`).
2. Higher-order similarity distribution moments (`StdSimilarityToSeen`, `AvgSimilarityToSeen`).
3. Cross-model ratios and latent factor embeddings (`SVD_Item_Latent`, `Ratio_IALS_SLIMElastic`).

---

## Artifacts in this Folder

* **[`xgb_optuna_trials.txt`](xgb_optuna_trials.txt)**: Raw execution log of the final 100-trial 5-Fold Optuna optimization.
* **[`feature_importance.png`](feature_importance.png)**: Visual ranking and relative importance distribution of the top features.
* **[`feature_importance_weights.txt`](feature_importance_weights.txt)**: Numerical importance weights assigned by the final XGBRanker model.
