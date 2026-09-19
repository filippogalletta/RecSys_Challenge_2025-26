#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gorgonzola Racing Team - RecSys Challenge 2025/26
Stage 2: XGBoost Reranker Wrapper & Submission Generator

Questo modulo incapsula il modello di Reranking (XGBRanker) e ne fornisce
un'interfaccia standard compatibile sia con l'Evaluator del corso che con
la procedura di inferenza per la Leaderboard.
"""

from typing import List, Union
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class XGBoostRerankerRecommender:
    """
    Recommender wrapper che applica il modello XGBoost addestrato per riordinare
    i candidati generati nello Stage 1.
    """

    RECOMMENDER_NAME = "XGBoostRerankerRecommender"

    def __init__(self, URM_train, XGB_model, df: pd.DataFrame):
        """
        :param URM_train: Matrice User-Rating di training
        :param XGB_model: Modello XGBRanker addestrato
        :param df: DataFrame contenente le feature per ciascuna coppia (UserID, ItemID) candidata
        """
        self.URM_train = URM_train
        self.XGB_model = XGB_model
        self.df = df

    def recommend(
        self, 
        user_ids: Union[int, List[int], np.ndarray], 
        cutoff: int = 20, 
        return_scores: bool = False, 
        remove_seen_flag: bool = True, 
        remove_top_pop_flag: bool = False, 
        remove_custom_items_flag: bool = False
    ):
        """
        Genera le top-cutoff raccomandazioni riordinate da XGBoost per uno o più utenti.
        """
        if isinstance(user_ids, (int, np.integer)):
            user_ids = [user_ids]

        recommendations = []
        for user_id in user_ids:
            df_slice = self.df[self.df['UserID'] == user_id]
            if len(df_slice) == 0:
                recommendations.append([])
                continue
                
            items = df_slice['ItemID'].to_numpy()
            
            # Seleziona solo le colonne feature escludendo UserID, ItemID e Label (se presenti)
            feature_cols = [c for c in df_slice.columns if c not in ['UserID', 'ItemID', 'Label']]
            preds = self.XGB_model.predict(df_slice[feature_cols])
            
            top_indices = np.argsort(preds)[-cutoff:][::-1]
            recommendations.append(items[top_indices].tolist())

        rec_array = np.array(recommendations, dtype=object)

        if return_scores:
            # Ritorna dummy scores per compatibilità con EvaluatorHoldout del framework
            return rec_array, None

        return rec_array

    def get_URM_train(self):
        return self.URM_train


def generate_submission(
    recommender: XGBoostRerankerRecommender, 
    target_user_ids: List[int], 
    output_path: str = "recommendations_xgboost.csv", 
    cutoff: int = 20
) -> pd.DataFrame:
    """
    Genera il file CSV di submission formattato secondo i requisiti della challenge:
    user_id,item_list
    0,2530 828 6411 ...
    """
    print(f"Generazione submission per {len(target_user_ids)} utenti target...")
    results = []
    
    for user_id in tqdm(target_user_ids, desc="Reranking predictions"):
        recs = recommender.recommend([user_id], cutoff=cutoff)
        item_str = " ".join(map(str, recs[0]))
        results.append((user_id, item_str))

    df_submission = pd.DataFrame(results, columns=["user_id", "item_list"])
    df_submission.to_csv(output_path, index=False)
    print(f"Submission salvata con successo in: {output_path}")
    return df_submission
