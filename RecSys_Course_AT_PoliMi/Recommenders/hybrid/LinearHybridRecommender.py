from Recommenders.BaseRecommender import BaseRecommender
import numpy as np

class GeneralizedLinearCoupleHybridRecommender(BaseRecommender):
    """
    This recommender MERGES two recommenders by weighting their RATINGS
    """

    RECOMMENDER_NAME = "GeneralizedLinearCoupleHybridRecommender"

    def __init__(self, URM_train, recommenders: list, verbose=True):
        self.RECOMMENDER_NAME = ''
        assert len(recommenders) == 2
        for recommender in recommenders:
            self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'HybridRecommender'

        super(GeneralizedLinearCoupleHybridRecommender, self).__init__(URM_train, verbose=verbose)

        self.recommenders = recommenders

    def fit(self, alpha=None):
        self.alpha = alpha

    def save_model(self, folder_path, file_name=None):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        result = self.alpha*self.recommenders[0]._compute_item_score(user_id_array,items_to_compute)
        result += (1-self.alpha)*self.recommenders[1]._compute_item_score(user_id_array,items_to_compute)
        return result

from Recommenders.BaseRecommender import BaseRecommender
import numpy as np

class NormalizedLinearCoupleHybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "NormalizedLinearCoupleHybridRecommender"

    def __init__(self, URM_train, recommenders: list, verbose=True):
        self.RECOMMENDER_NAME = ''
        assert len(recommenders) == 2
        for recommender in recommenders:
            self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'HybridRecommender'

        # CORREZIONE: super() deve puntare alla classe corrente o essere usato senza argomenti
        super(NormalizedLinearCoupleHybridRecommender, self).__init__(URM_train, verbose=verbose)
        self.recommenders = recommenders

    def fit(self, alpha=0.5):
        self.alpha = alpha

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # 1. Ottieni gli score originali
        score_0 = self.recommenders[0]._compute_item_score(user_id_array, items_to_compute)
        score_1 = self.recommenders[1]._compute_item_score(user_id_array, items_to_compute)

        # 2. STANDARDIZATION (Z-Score) come da Practice 12
        # Sottraiamo la media e dividiamo per std per rendere gli score comparabili
        mean_0 = np.mean(score_0, axis=1, keepdims=True)
        std_0 = np.std(score_0, axis=1, keepdims=True) + 1e-9
        score_0 = (score_0 - mean_0) / std_0

        mean_1 = np.mean(score_1, axis=1, keepdims=True)
        std_1 = np.std(score_1, axis=1, keepdims=True) + 1e-9
        score_1 = (score_1 - mean_1) / std_1

        # 3. Combinazione pesata
        result = self.alpha * score_0 + (1 - self.alpha) * score_1
        return result
        