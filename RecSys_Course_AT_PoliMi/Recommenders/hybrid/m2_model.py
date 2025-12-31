from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from scipy.sparse import csr_matrix

class IntegratedHierarchicalHybridRecommender(BaseRecommender):
    """
    Questo recommender gestisce una gerarchia di ibridazione:
    1. Crea una similarità ibrida combinando EASE e SLIM (controllato da alpha).
    2. Istanzia un ItemKNNCustomSimilarityRecommender con quella similarità.
    3. Esegue una somma pesata degli score tra il CustomKNN e IALS (controllato da beta).
    """

    RECOMMENDER_NAME = "IntegratedHierarchicalHybridRecommender"

    def __init__(self, URM_train, rec_sim_1, rec_sim_2, rec_lin_3, verbose=True):
        super(IntegratedHierarchicalHybridRecommender, self).__init__(URM_train, verbose=verbose)
        
        # Salviamo i modelli già addestrati
        self.rec_sim_1 = rec_sim_1  # EASE
        self.rec_sim_2 = rec_sim_2  # SLIM
        self.rec_lin_3 = rec_lin_3  # IALS
        
        self.rec_custom_knn = None
        self.alpha = None
        self.beta = None

    def fit(self, alpha=0.5, beta=0.5):
        """
        alpha: Peso per la combinazione delle similarità (0 <= alpha <= 1)
               New_W = (1 - alpha) * EASE_W + alpha * SLIM_W
        beta: Peso per la combinazione degli score (0 <= beta <= 1)
              Final_Score = beta * CustomKNN_Score + (1 - beta) * IALS_Score
        """
        self.alpha = alpha
        self.beta = beta
        
        # --- STEP 1: Combinazione delle Matrici di Similarità ---
        if self.verbose:
            print(f"{self.RECOMMENDER_NAME}: Creating hybrid similarity with alpha={self.alpha}...")
            
        # Nota: Assumiamo che rec_ease e rec_slim abbiano l'attributo W_sparse
        # Seguo la tua logica: (1 - alpha) * EASE + alpha * SLIM
        new_similarity = (1 - self.alpha) * csr_matrix(self.rec_sim_1.W_sparse) + \
                         self.alpha * csr_matrix(self.rec_sim_2.W_sparse)
        
        # --- STEP 2: Setup del Recommender Intermedio ---
        self.rec_custom_knn = ItemKNNCustomSimilarityRecommender(self.URM_train)
        self.rec_custom_knn.fit(new_similarity)
        
        if self.verbose:
            print(f"{self.RECOMMENDER_NAME}: Fitting complete. Ready to score with beta={self.beta}.")

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # --- STEP 3: Calcolo e Combinazione degli Score ---
        
        # 1. Ottieni gli score dal modello basato sulla similarità ibrida
        item_weights_1 = self.rec_custom_knn._compute_item_score(user_id_array, items_to_compute)
        
        # 2. Ottieni gli score da IALS
        item_weights_2 = self.rec_lin_3._compute_item_score(user_id_array, items_to_compute)

        # 3. Combina linearmente i risultati usando beta
        # Logica: beta * (EASE+SLIM) + (1-beta) * IALS
        result = self.beta * item_weights_1 + (1 - self.beta) * item_weights_2

        return result