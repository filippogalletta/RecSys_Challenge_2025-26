from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from scipy.sparse import csr_matrix
import numpy as np

class TripleIntegratedHierarchicalHybridRecommender(BaseRecommender):
    """
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
        alpha: peso di rec_sim_2
        beta: peso di rec_sim_3
        gamma: bilanciamento tra l'ibrido di similarità e rec_score_4 (peso del 4o modello)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        """if self.verbose:
            print(f"{self.RECOMMENDER_NAME}: Fusing similarity matrices...")"""

        # --- STEP 1: Merge delle prime due matrici (es. SLIM e EASE) ---
        w1 = csr_matrix(self.rec_sim_1.W_sparse)
        w2 = csr_matrix(self.rec_sim_2.W_sparse)
        w_12 = (1 - self.alpha) * w1 + self.alpha * w2

        # --- STEP 2: Merge con la terza matrice (es. RP3Beta) ---
        w3 = csr_matrix(self.rec_sim_3.W_sparse)
        new_similarity = (1 - self.beta) * w_12 + self.beta * w3
        
        # --- STEP 3: Setup del Recommender di similarità fuso ---
        self.rec_custom_knn = ItemKNNCustomSimilarityRecommender(self.URM_train)
        self.rec_custom_knn.fit(new_similarity)
        
        """if self.verbose:
            print(f"{self.RECOMMENDER_NAME}: Similarity fusion complete. Final score combination with gamma={self.gamma}.")"""

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # 1. Score dall'ibrido di similarità (SLIM + EASE + RP3)
        item_weights_sim = self.rec_custom_knn._compute_item_score(user_id_array, items_to_compute)
        
        # 2. Score dal modello lineare (iALS)
        item_weights_score = self.rec_score_4._compute_item_score(user_id_array, items_to_compute)
        
        # --- STEP 4: Combinazione lineare finale degli score ---
        # (1 - gamma) * Similarity + gamma * iALS
        item_weights = (1 - self.gamma) * item_weights_sim + self.gamma * item_weights_score
        
        return item_weights

    def save_model(self, folder_path, file_name=None):
        # Salvataggio del solo recommender fuso per efficienza
        if self.rec_custom_knn is not None:
            self.rec_custom_knn.save_model(folder_path, file_name=file_name)