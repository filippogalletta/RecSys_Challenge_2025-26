import numpy as np

class XGBoostRerankerRecommender:
    def __init__(self, URM_train, XGB_model, df):
        self.URM_train = URM_train
        self.df = df
        self.XGB_model = XGB_model

    def recommend(self, user_ids, cutoff=20, return_scores=True, remove_seen_flag=True, remove_top_pop_flag=True, remove_custom_items_flag=False):
        recommendations = []
        for user_id in user_ids:
            # print(user_id)
            df_slice = self.df[self.df['UserID'] == user_id]
            items = df_slice.ItemID.to_numpy()
            preds = self.XGB_model.predict(df_slice)
            recommendations.append(items[np.argsort(preds)[-cutoff:][::-1]].tolist())
        
        if return_scores:
            rec, scores = 0
            # useless scores
            return np.array(recommendations), scores
        
        return np.array(recommendations)

    def get_URM_train(self):
        return self.URM_train