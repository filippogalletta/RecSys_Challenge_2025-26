from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from scipy.sparse import csr_matrix

class SimilarityMergingHybridRecommender(BaseRecommender):
    """
    Questo recommender fonde tre modelli similarity-based (es. EASE, SLIM, KNN)
    sommando le loro W_sparse in un'unica matrice di similarità
    """

    RECOMMENDER_NAME = "SimilarityMergingHybridRecommender"

    def __init__(self, URM_train, rec_1, rec_2, rec_3, verbose=True):
        super(SimilarityMergingHybridRecommender, self).__init__(URM_train, verbose=verbose)
        
        self.rec_1 = rec_1  # EASE
        self.rec_2 = rec_2  # SLIM
        self.rec_3 = rec_3  # KNN

    def fit(self, alpha=0.5, beta=0.5):
        """
        alpha: peso tra il modello 1 e 2
        beta: peso tra l'ibrido (1+2) e il modello 3
        """
        self.alpha = alpha
        self.beta = beta
        
        # 1. Estrazione e conversione delle matrici W_sparse
        w1 = csr_matrix(self.rec_1.W_sparse)
        w2 = csr_matrix(self.rec_2.W_sparse)
        w3 = csr_matrix(self.rec_3.W_sparse)

        # 2. Triple Similarity Merging (Logica gerarchica) 
        w_12 = w1 * (1 - self.alpha) + w2 * self.alpha
        # Prima fondiamo i primi due, poi il risultato con il terzo
        new_similarity = w_12 * (1 - self.beta) + w3 * self.beta

        # 3. Creazione del Recommender finale
        self.rec_final = ItemKNNCustomSimilarityRecommender(self.URM_train)
        self.rec_final.fit(new_similarity)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Una volta fusa la matrice, usiamo solo il recommender finale.
        # È molto più veloce perché esegue una sola moltiplicazione URM * S_ibrida 
        return self.rec_final._compute_item_score(user_id_array, items_to_compute)

    def save_model(self, folder_path, file_name=None):
        self.rec_final.save_model(folder_path, file_name=file_name)