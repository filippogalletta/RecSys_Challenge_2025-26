#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gorgonzola Racing Team - RecSys Challenge 2025/26
Stage 1: Candidate Generation (Hierarchical Hybrid Recommenders)

Contiene:
- TripleIntegratedHierarchicalHybridRecommender (M3):
  1. Fusione matrici di similarità (W_sparse): Modello 1 (SLIM) + Modello 2 (EASE) -> alpha
  2. Fusione matrici di similarità (W_sparse): Risultato (1+2) + Modello 3 (RP3beta) -> beta
  3. Ibridazione lineare degli score: Risultato Similarità + Modello 4 (iALS) -> gamma
- IntegratedHierarchicalHybridRecommender (M2):
  1. Fusione matrici di similarità (W_sparse): EASE + SLIM -> alpha
  2. Combinazione lineare degli score: CustomKNN + iALS -> beta
"""

from scipy.sparse import csr_matrix
import numpy as np

try:
    from Recommenders.BaseRecommender import BaseRecommender
    from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
except ImportError:
    try:
        from RecSys_Course_AT_PoliMi.Recommenders.BaseRecommender import BaseRecommender
        from RecSys_Course_AT_PoliMi.Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
    except ImportError:
        BaseRecommender = object
        ItemKNNCustomSimilarityRecommender = None


class TripleIntegratedHierarchicalHybridRecommender(BaseRecommender):
    """
    Hierarchical Hybrid Recommender (M3 Model):
    1. Ibridazione Similarity (W_sparse): Modello 1 (SLIM) + Modello 2 (EASE) -> alpha
    2. Ibridazione Similarity (W_sparse): Risultato (1+2) + Modello 3 (RP3Beta) -> beta
    3. Ibridazione Score: Risultato Similarity + Modello 4 (iALS) -> gamma
    """

    RECOMMENDER_NAME = "TripleIntegratedHierarchicalHybridRecommender"

    def __init__(self, URM_train, rec_sim_1, rec_sim_2, rec_sim_3, rec_score_4, verbose=True):
        super(TripleIntegratedHierarchicalHybridRecommender, self).__init__(URM_train, verbose=verbose)
        
        # Modelli per il merging della similarità (es. SLIM, EASE, RP3Beta)
        self.rec_sim_1 = rec_sim_1
        self.rec_sim_2 = rec_sim_2
        self.rec_sim_3 = rec_sim_3
        
        # Modello per l'ibridazione lineare degli score (es. iALS)
        self.rec_score_4 = rec_score_4
        
        self.rec_custom_knn = None
        self.alpha = None
        self.beta = None
        self.gamma = None

    def fit(self, alpha=0.5, beta=0.5, gamma=0.5):
        """
        :param alpha: peso di rec_sim_2 (EASE) rispetto a rec_sim_1 (SLIM)
        :param beta: peso di rec_sim_3 (RP3Beta) rispetto all'ibrido precedente
        :param gamma: bilanciamento tra l'ibrido di similarità e rec_score_4 (iALS)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # --- STEP 1: Merge delle prime due matrici (es. SLIM ed EASE) ---
        w1 = csr_matrix(self.rec_sim_1.W_sparse)
        w2 = csr_matrix(self.rec_sim_2.W_sparse)
        w_12 = (1.0 - self.alpha) * w1 + self.alpha * w2

        # --- STEP 2: Merge con la terza matrice (es. RP3Beta) ---
        w3 = csr_matrix(self.rec_sim_3.W_sparse)
        new_similarity = (1.0 - self.beta) * w_12 + self.beta * w3
        
        # --- STEP 3: Setup del Recommender di similarità fuso ---
        if ItemKNNCustomSimilarityRecommender is None:
            raise ImportError("ItemKNNCustomSimilarityRecommender non trovato. Assicurati che RecSys_Course_AT_PoliMi sia nel PYTHONPATH.")
        
        self.rec_custom_knn = ItemKNNCustomSimilarityRecommender(self.URM_train)
        self.rec_custom_knn.fit(new_similarity)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # 1. Score dall'ibrido di similarità (SLIM + EASE + RP3)
        item_weights_sim = self.rec_custom_knn._compute_item_score(user_id_array, items_to_compute)
        
        # 2. Score dal modello a fattori latenti (iALS)
        item_weights_score = self.rec_score_4._compute_item_score(user_id_array, items_to_compute)
        
        # --- STEP 4: Combinazione lineare finale degli score ---
        # (1 - gamma) * Similarity + gamma * iALS
        item_weights = (1.0 - self.gamma) * item_weights_sim + self.gamma * item_weights_score
        
        return item_weights

    def save_model(self, folder_path, file_name=None):
        if self.rec_custom_knn is not None:
            self.rec_custom_knn.save_model(folder_path, file_name=file_name)


class IntegratedHierarchicalHybridRecommender(BaseRecommender):
    """
    Hierarchical Hybrid Recommender (M2 Model):
    1. Crea una similarità ibrida combinando EASE e SLIM (controllato da alpha).
    2. Istanzia un ItemKNNCustomSimilarityRecommender con quella similarità.
    3. Esegue una somma pesata degli score tra il CustomKNN e IALS (controllato da beta).
    """

    RECOMMENDER_NAME = "IntegratedHierarchicalHybridRecommender"

    def __init__(self, URM_train, rec_sim_1, rec_sim_2, rec_lin_3, verbose=True):
        super(IntegratedHierarchicalHybridRecommender, self).__init__(URM_train, verbose=verbose)
        
        self.rec_sim_1 = rec_sim_1  # EASE
        self.rec_sim_2 = rec_sim_2  # SLIM
        self.rec_lin_3 = rec_lin_3  # IALS
        
        self.rec_custom_knn = None
        self.alpha = None
        self.beta = None

    def fit(self, alpha=0.5, beta=0.5):
        """
        alpha: peso combinazione similarità (1 - alpha) * EASE + alpha * SLIM
        beta: peso combinazione score beta * CustomKNN + (1 - beta) * IALS
        """
        self.alpha = alpha
        self.beta = beta
        
        new_similarity = (1.0 - self.alpha) * csr_matrix(self.rec_sim_1.W_sparse) + \
                         self.alpha * csr_matrix(self.rec_sim_2.W_sparse)
        
        if ItemKNNCustomSimilarityRecommender is None:
            raise ImportError("ItemKNNCustomSimilarityRecommender non trovato. Assicurati che RecSys_Course_AT_PoliMi sia nel PYTHONPATH.")

        self.rec_custom_knn = ItemKNNCustomSimilarityRecommender(self.URM_train)
        self.rec_custom_knn.fit(new_similarity)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.rec_custom_knn._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.rec_lin_3._compute_item_score(user_id_array, items_to_compute)
        return self.beta * item_weights_1 + (1.0 - self.beta) * item_weights_2
