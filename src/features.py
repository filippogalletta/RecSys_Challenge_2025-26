#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gorgonzola Racing Team - RecSys Challenge 2025/26
Stage 2: Feature Engineering & Tabular Dataset Construction

Questo modulo genera 70+ feature tabular per ogni coppia (UserID, ItemID) candidata:
1. Punteggi dei modelli base (L_inf normalized)
2. Score standardizzati per utente (Z-score normalizzato: Score_norm)
3. Posizioni in classifica (RankPosition) e rank logaritmici inversi (RankInv)
4. Momenti statistici di similarità verso gli item visti (Avg, Max, Min, Std, Skewness, Kurtosis)
   per SLIMElastic, RP3beta e ItemKNNCF
5. Statistiche cross-modello sui rank (Mean, Std, Skewness, Kurtosis, Variance tra rank dei modelli)
6. Feature di popolarità globale e mainstreamness di utenti e item (distribuzioni e delta)
7. Embedding latenti utente/item (prime 5 componenti) estratti da IALS e ScaledPureSVD
8. Feature comparative cross-modello: Score Ratios e Rank Differences tra modelli a fattori latenti
   e modelli a neighborhood/grafo
"""

import gc
from typing import Any, Dict, Union
import numpy as np
from numpy import linalg as LA
import pandas as pd
import scipy.sparse as sp
import scipy.sparse as sps
import scipy.stats as stats
from tqdm.auto import tqdm

# Aliases
RecommenderModel = Any
URM_Type = Union[np.ndarray, sps.csr_matrix]


def optimize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ottimizzazione della memoria RAM del DataFrame tramite downcasting dei tipi numerici.
    """
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def feature_populator(
    URM: URM_Type, 
    candidate_generator: RecommenderModel, 
    other_algorithms: Dict[str, RecommenderModel], 
    cutoff: int = 90
) -> pd.DataFrame:
    """
    Costruisce il DataFrame di feature per XGBoost.
    
    :param URM: Matrice User-Rating (CSR o ndarray)
    :param candidate_generator: Modello usato per generare i candidati (es. M3)
    :param other_algorithms: Dizionario {nome: modello} con tutti i recommender base addestrati
    :param cutoff: Numero di candidati per utente
    :return: DataFrame pronto per il ranking, contenente UserID, ItemID e tutte le feature
    """
    n_users, n_items = URM.shape

    # 1. Inizializzazione DataFrame
    print("--- Step 1/7: Inizializzazione struttura dati ---")
    training_dataframe = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
    training_dataframe.index.name = "UserID"

    # 2. Generazione candidati
    print(f"--- Step 2/7: Retrieval di {cutoff} candidati per ciascun utente ---")
    for user_id in tqdm(range(n_users), desc="Candidate Generation"):  
        recommendations = candidate_generator.recommend(user_id, cutoff=cutoff)
        training_dataframe.loc[user_id, "ItemID"] = recommendations  

    training_dataframe = training_dataframe.explode("ItemID")

    # 3. Estrazione Score, Rank e Normalizzazioni dai Modelli Base
    print("--- Step 3/7: Calcolo Score Linf, Z-Score, Rank e RankInv per ciascun modello base ---")
    for algorithm_name, recommender in tqdm(other_algorithms.items(), desc="Base Model Signals"):
        scores = recommender._compute_item_score(np.arange(n_users))
        norm_linf_scores = scores / (LA.norm(scores, np.inf, axis=1, keepdims=True) + 1e-6)

        for user_id in tqdm(range(n_users), desc=f"Scoring {algorithm_name}", leave=False):
            item_list = training_dataframe.loc[user_id, "ItemID"].values.tolist()
            norm_linf_scores[user_id, :] = recommender._remove_seen_on_scores(user_id, norm_linf_scores[user_id, :])
            training_dataframe.loc[user_id, f"{algorithm_name}_Score"] = norm_linf_scores[user_id, item_list]
            
            # Normalizzazione Z-score per utente
            candidate_scores = norm_linf_scores[user_id, item_list]
            candidate_scores = np.nan_to_num(candidate_scores, nan=0.0, posinf=0.0, neginf=0.0)
            mean_u = candidate_scores.mean()
            std_u = candidate_scores.std() + 1e-6

            candidate_scores_norm = (candidate_scores - mean_u) / std_u
            training_dataframe.loc[user_id, f"{algorithm_name}_Score_norm"] = candidate_scores_norm
            
            # Rank e Rank Logaritmico Inverso
            rank = np.argsort(norm_linf_scores[user_id, :])[::-1]
            positions = np.zeros(n_items, dtype=int)
            positions[rank] = np.arange(n_items)
            
            rank_pos = positions[item_list]
            rank_inv = 1.0 / np.log2(2 + rank_pos)
            training_dataframe.loc[user_id, f"{algorithm_name}_RankInv"] = rank_inv
            training_dataframe.loc[user_id, f"{algorithm_name}_RankPosition"] = positions[item_list]

        del scores, norm_linf_scores, rank, positions
        gc.collect()

    # 4. Momenti Statistici di Similarità verso gli Item Visti
    print("--- Step 4/7: Calcolo statistiche di similarità (Mean, Max, Min, Std, Skewness, Kurtosis) ---")
    similarity_models = {
        'SLIMElastic': 'SLIMElastic',
        'RP3beta': 'RP3',
        'ItemKNNCF': 'ItemKNNCF'
    }

    for algo_name, suffix in similarity_models.items():
        if algo_name not in other_algorithms:
            print(f"Skipping {algo_name}: modello non presente in other_algorithms.")
            continue
            
        print(f"Calcolo momenti similarità per {algo_name}...")
        item_item_S = other_algorithms[algo_name].W_sparse.toarray()
        
        col_names = {
            'avg': f"AvgSimilarityToSeen{suffix}",
            'max': f"MaxSimilarityToSeen{suffix}",
            'min': f"MinSimilarityToSeen{suffix}",
            'std': f"StdSimilarityToSeen{suffix}",
            'skew': f"SkewSimilarityToSeen{suffix}",
            'kurtosis': f"KurtosisSimilarityToSeen{suffix}"
        }

        for user_id in tqdm(range(n_users), desc=f"Users {algo_name}", leave=False):
            seen_items = URM[user_id].nonzero()[1] 
            
            if len(seen_items) == 0:
                training_dataframe.loc[user_id, col_names['avg']] = 0
                training_dataframe.loc[user_id, col_names['max']] = 0
                training_dataframe.loc[user_id, col_names['min']] = 0
                training_dataframe.loc[user_id, col_names['std']] = 0
                training_dataframe.loc[user_id, col_names['skew']] = 0
                training_dataframe.loc[user_id, col_names['kurtosis']] = 0
            else:
                candidate_items = training_dataframe.loc[user_id, "ItemID"].values.astype(int)
                similarities = item_item_S[candidate_items, :][:, seen_items]
                
                training_dataframe.loc[user_id, col_names['avg']] = similarities.mean(axis=1).flatten()
                training_dataframe.loc[user_id, col_names['max']] = similarities.max(axis=1).flatten()
                training_dataframe.loc[user_id, col_names['min']] = similarities.min(axis=1).flatten()
                training_dataframe.loc[user_id, col_names['std']] = similarities.std(axis=1).flatten()
                training_dataframe.loc[user_id, col_names['skew']] = stats.skew(similarities, axis=1)
                training_dataframe.loc[user_id, col_names['kurtosis']] = stats.kurtosis(similarities, axis=1)

        del item_item_S
        gc.collect()

    # 5. Statistiche Cross-Modello sui Rank
    print("--- Step 5/7: Calcolo statistiche cross-modello e concordanza rank ---")
    recommended_columns = [col for col in training_dataframe.columns if col.endswith('_Recommended')]
    if recommended_columns:
        training_dataframe['Counter_Recommended'] = training_dataframe[recommended_columns].sum(axis=1).astype(int)

    position_columns = [col for col in training_dataframe.columns if col.endswith('_RankPosition')]
    training_dataframe['Mean_RankPosition'] = training_dataframe[position_columns].mean(axis=1)
    training_dataframe['Std_RankPosition'] = training_dataframe[position_columns].std(axis=1)
    training_dataframe['Skew_RankPosition'] = training_dataframe[position_columns].skew(axis=1)
    training_dataframe['Kurtosis_RankPosition'] = training_dataframe[position_columns].kurtosis(axis=1)

    training_dataframe = training_dataframe.reset_index()
    training_dataframe = training_dataframe.rename(columns={"index": "UserID"})

    # 6. Popolarità, Lunghezza Profilo e Mainstreamness
    print("--- Step 6/7: Calcolo metriche di popolarità e mainstreamness ---")
    item_popularity = np.ediff1d(sps.csc_matrix(URM).indptr)
    item_popularity = item_popularity / np.max(item_popularity)
    training_dataframe['item_popularity'] = item_popularity[training_dataframe["ItemID"].values.astype(int)]

    user_popularity = np.ediff1d(sps.csr_matrix(URM).indptr)
    user_popularity = user_popularity / np.max(user_popularity)
    training_dataframe['user_profile_len'] = user_popularity[training_dataframe["UserID"].values.astype(int)]

    mainstream_user = URM.dot(item_popularity) / np.where(user_popularity == 0, 1, user_popularity)
    training_dataframe['mainstream_user'] = mainstream_user[training_dataframe["UserID"].values.astype(int)]

    mainstream_item = URM.T.dot(user_popularity) / np.where(item_popularity == 0, 1, item_popularity)
    training_dataframe['mainstream_item'] = mainstream_item[training_dataframe["ItemID"].values.astype(int)]

    # Embedding IALS (primi 5 fattori)
    if 'IALS' in other_algorithms:
        u_factors = other_algorithms['IALS'].USER_factors
        i_factors = other_algorithms['IALS'].ITEM_factors
        n_latent = 5 

        u_cols = [f"IALS_User_Latent_{i}" for i in range(n_latent)]
        i_cols = [f"IALS_Item_Latent_{i}" for i in range(n_latent)]

        user_factors_df = pd.DataFrame(u_factors[:, :n_latent], columns=u_cols)
        user_factors_df['UserID'] = np.arange(n_users)
        training_dataframe = pd.merge(training_dataframe, user_factors_df, on='UserID', how='left')

        item_factors_df = pd.DataFrame(i_factors[:, :n_latent], columns=i_cols)
        item_factors_df['ItemID'] = np.arange(n_items)
        training_dataframe = pd.merge(training_dataframe, item_factors_df, on='ItemID', how='left')

    # Rank Variance (Disaccordo tra i modelli)
    rank_cols = [c for c in training_dataframe.columns if c.endswith('_RankPosition')]
    training_dataframe['Rank_Variance'] = training_dataframe[rank_cols].var(axis=1)

    # Popolarità media e deviazione standard per utente
    item_pop = np.ediff1d(sps.csc_matrix(URM).indptr)
    item_pop = item_pop / item_pop.max()

    user_pop_sum = URM.dot(item_pop)
    user_profile_len = np.ediff1d(sps.csr_matrix(URM).indptr)
    user_profile_len[user_profile_len == 0] = 1 
    user_avg_pop = user_pop_sum / user_profile_len

    user_pop_sq_sum = URM.dot(item_pop ** 2)
    user_avg_pop_sq = user_pop_sq_sum / user_profile_len
    user_std_pop = np.sqrt(np.maximum(0, user_avg_pop_sq - user_avg_pop**2))

    user_features = pd.DataFrame({
        'UserID': np.arange(n_users),
        'User_Avg_Item_Popularity': user_avg_pop,
        'User_Std_Item_Popularity': user_std_pop
    })
    training_dataframe = pd.merge(training_dataframe, user_features, on='UserID', how='left')

    if 'item_popularity' not in training_dataframe.columns:
        training_dataframe['item_popularity'] = item_pop[training_dataframe['ItemID'].values.astype(int)]
        
    training_dataframe['Pop_Diff_Item_UserAvg'] = training_dataframe['item_popularity'] - training_dataframe['User_Avg_Item_Popularity']

    # Embedding ScaledPureSVD (primi 5 fattori)
    if 'ScaledPureSVD' in other_algorithms:
        svd_model = other_algorithms['ScaledPureSVD']
        u_factors_svd = svd_model.USER_factors
        i_factors_svd = svd_model.ITEM_factors
        n_latent_svd = 5
        
        u_cols_svd = [f"SVD_User_Latent_{i}" for i in range(n_latent_svd)]
        i_cols_svd = [f"SVD_Item_Latent_{i}" for i in range(n_latent_svd)]
        
        user_factors_svd_df = pd.DataFrame(u_factors_svd[:, :n_latent_svd], columns=u_cols_svd)
        user_factors_svd_df['UserID'] = np.arange(n_users)
        training_dataframe = pd.merge(training_dataframe, user_factors_svd_df, on='UserID', how='left')
        
        item_factors_svd_df = pd.DataFrame(i_factors_svd[:, :n_latent_svd], columns=i_cols_svd)
        item_factors_svd_df['ItemID'] = np.arange(n_items)
        training_dataframe = pd.merge(training_dataframe, item_factors_svd_df, on='ItemID', how='left')
        
        del user_factors_svd_df, item_factors_svd_df
        gc.collect()

    # 7. Feature Comparative: Score Ratios & Rank Differences
    print("--- Step 7/7: Calcolo Score Ratios & Rank Differences cross-modello ---")
    model_pairs = [
        ('IALS', 'SLIMElastic'),          # Latent MF vs Sparse Linear
        ('IALS', 'RP3beta'),              # Latent MF vs Graph
        ('ScaledPureSVD', 'SLIMElastic'), # Linear MF vs Sparse Linear
        ('ScaledPureSVD', 'RP3beta'),     # Linear MF vs Graph
        ('IALS', 'ItemKNNCF'),            # Latent MF vs Item Similarity
        ('SLIMElastic', 'ItemKNNCF'),     # Sparse Linear vs Item Similarity
    ]

    eps = 1e-6
    for algo_a, algo_b in model_pairs:
        col_score_a = f"{algo_a}_Score"
        col_score_b = f"{algo_b}_Score"
        
        if col_score_a in training_dataframe.columns and col_score_b in training_dataframe.columns:
            # Score Ratio
            training_dataframe[f"Ratio_{algo_a}_{algo_b}"] = (training_dataframe[col_score_a] + eps) / (training_dataframe[col_score_b] + eps)
            # Rank Difference
            training_dataframe[f"Diff_{algo_a}_{algo_b}"] = training_dataframe[f"{algo_a}_RankPosition"] - training_dataframe[f"{algo_b}_RankPosition"]

    gc.collect()
    print(f"DataFrame completato: {training_dataframe.shape[0]} righe, {training_dataframe.shape[1]} feature generate con successo.")
    return training_dataframe
