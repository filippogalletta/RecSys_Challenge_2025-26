import numpy as np
import scipy.sparse as sps
import scipy.sparse as sp
from scipy.sparse import csr_matrix, issparse
from typing import List, Dict, Any, Tuple, Union
from tqdm import tqdm
from tqdm.auto import tqdm
import pandas as pd
from numpy import linalg as LA
import gc
import scipy.stats as stats

# Definizione degli Alias
RecommenderModel = Any
URM_Type = Union[np.ndarray, csr_matrix]



# feature_populator(URM, linear_comb_rec, other_algorithms, cutoff = 50

def feature_populator(
    URM: URM_Type, 
    candidate_generator: RecommenderModel, 
    other_algorithms: dict[str, RecommenderModel], # dizionario other_algorithms
    cutoff: int,
    ):
    
    cutoff = 50

    # feature_populator(URM, linear_comb_rec, other_algorithms, cutoff = 50
    
    #1
    print('---------------1---------------')
    n_users, n_items = URM.shape
    training_dataframe = pd.DataFrame(index=range(0,n_users), columns = ["ItemID"])
    training_dataframe.index.name='UserID'

    #2
    print('---------------2---------------')
    for user_id in tqdm(range(n_users)):  
        recommendations = candidate_generator.recommend(user_id, cutoff = cutoff)
        training_dataframe.loc[user_id, "ItemID"] = recommendations  

    training_dataframe = training_dataframe.explode("ItemID")

    #3 
    print('---------------3---------------')

    for algorithm_name, recommender in tqdm(other_algorithms.items()):
        scores = recommender._compute_item_score(np.arange(n_users))
        norm_linf_scores = scores / (LA.norm(scores, np.inf, axis=1, keepdims=True) + 1e-6)

        for user_id in tqdm(range(n_users)):
            item_list = training_dataframe.loc[user_id, "ItemID"].values.tolist()
            norm_linf_scores[user_id, :] = recommender._remove_seen_on_scores(user_id, norm_linf_scores[user_id, :])
            training_dataframe.loc[user_id, f"{algorithm_name}_Score"] = norm_linf_scores[user_id, item_list]

            rank = np.argsort(norm_linf_scores[user_id, :])[::-1]
            positions = np.zeros(n_items, dtype=int)
            positions[rank] = np.arange(n_items)
            training_dataframe.loc[user_id, f"{algorithm_name}_RankPosition"] = positions[item_list]

        del scores, norm_linf_scores, rank, positions
        gc.collect()

# 4 & 5 Unificati
    print('--------------- 4 ---------------')

    # Dizionario: {'NomeChiaveNelDict': 'SuffissoColonna'}
    similarity_models = {
        'SLIMElastic': 'SLIMElastic',
        'RP3beta': 'RP3',
        # Puoi aggiungere qui 'ItemKNNCF': 'ItemKNN' se vuoi
        'ItemKNNCF': 'ItemKNNCF'
    }

    for algo_name, suffix in similarity_models.items():
        if algo_name not in other_algorithms:
            print(f"Skipping {algo_name}: model not found.")
            continue
            
        print(f"Processing {algo_name}...")
        
        item_item_S = other_algorithms[algo_name].W_sparse.toarray()
        
        col_names = {
            'avg': f"AvgSimilarityToSeen{suffix}",
            'max': f"MaxSimilarityToSeen{suffix}",
            'min': f"MinSimilarityToSeen{suffix}",
            'std': f"StdSimilarityToSeen{suffix}",
            'skew': f"SkewSimilarityToSeen{suffix}",
            'kurtosis': f"KurtosisSimilarityToSeen{suffix}"
        }

        # Iterazione utenti
        for user_id in tqdm(range(n_users), desc=f"Users {algo_name}"):
            seen_items = URM[user_id].nonzero()[1] 
            
            if len(seen_items) == 0:
                training_dataframe.loc[user_id, col_names['avg']] = 0
                training_dataframe.loc[user_id, col_names['max']] = 0
                training_dataframe.loc[user_id, col_names['min']] = 0
                training_dataframe.loc[user_id, col_names['std']] = 0
                training_dataframe.loc[user_id, col_names['skew']] = 0
                training_dataframe.loc[user_id, col_names['kurtosis']] = 0
            else:
                # Estrazione candidati per l'utente corrente
                candidate_items = training_dataframe.loc[user_id, "ItemID"].values.astype(int)
                
                # Slicing della matrice
                similarities = item_item_S[candidate_items, :][:, seen_items]
                
                # Calcolo statistiche
                training_dataframe.loc[user_id, col_names['avg']] = similarities.mean(axis=1).flatten()
                training_dataframe.loc[user_id, col_names['max']] = similarities.max(axis=1).flatten()
                training_dataframe.loc[user_id, col_names['min']] = similarities.min(axis=1).flatten()
                training_dataframe.loc[user_id, col_names['std']] = similarities.std(axis=1).flatten()
                training_dataframe.loc[user_id, col_names['skew']] = stats.skew(similarities, axis=1)
                training_dataframe.loc[user_id, col_names['kurtosis']] = stats.kurtosis(similarities, axis=1)

        # Pulizia memoria fondamentale dentro il loop
        del item_item_S
        gc.collect()

    #6
    print('---------------6---------------')
    recommended_columns = [col for col in training_dataframe.columns if col.endswith('_Recommended')]
    training_dataframe['Counter_Recommended'] = training_dataframe[recommended_columns].sum(axis=1).astype(int)

    position_columns = [col for col in training_dataframe.columns if col.endswith('_RankPosition')]
    training_dataframe['Mean_RankPosition'] = training_dataframe[position_columns].mean(axis=1)
    training_dataframe['Std_RankPosition'] = training_dataframe[position_columns].std(axis=1)
    training_dataframe['Skew_RankPosition'] = training_dataframe[position_columns].skew(axis=1)
    training_dataframe['Kurtosis_RankPosition'] = training_dataframe[position_columns].kurtosis(axis=1)

    training_dataframe = training_dataframe.reset_index()
    training_dataframe = training_dataframe.rename(columns = {"index": "UserID"})


    #7
    print('---------------7---------------')
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
    training_dataframe

    # ------------- 5 ------------
    print('--------------- 5 ---------------')
    u_factors = other_algorithms['IALS'].USER_factors
    i_factors = other_algorithms['IALS'].ITEM_factors

    # Prendiamo le prime n dimensioni 
    n_latent = 5 

    u_cols = [f"IALS_User_Latent_{i}" for i in range(n_latent)]
    i_cols = [f"IALS_Item_Latent_{i}" for i in range(n_latent)]

    # User Factors
    user_factors_df = pd.DataFrame(u_factors[:, :n_latent], columns=u_cols)
    user_factors_df['UserID'] = np.arange(n_users)
    training_dataframe = pd.merge(training_dataframe, user_factors_df, on='UserID', how='left')

    # Item Factors
    item_factors_df = pd.DataFrame(i_factors[:, :n_latent], columns=i_cols)
    item_factors_df['ItemID'] = np.arange(n_items)
    training_dataframe = pd.merge(training_dataframe, item_factors_df, on='ItemID', how='left')

    # Seleziona le colonne dei rank
    rank_cols = [c for c in training_dataframe.columns if c.endswith('_RankPosition')]
    score_cols = [c for c in training_dataframe.columns if c.endswith('_Score')]

    # Varianza dei Rank (Disaccordo tra i modelli)
    training_dataframe['Rank_Variance'] = training_dataframe[rank_cols].var(axis=1)

    # Calcolo Popolarità Globale degli Item
    item_pop = np.ediff1d(sps.csc_matrix(URM).indptr)
    item_pop = item_pop / item_pop.max() # Normalizzazione

    # Vettorializzazione: Calcolo media e varianza popolarità per utente usando algebra lineare
    # più veloce di un for

    # 1. Somma delle popolarità degli item visti dall'utente
    user_pop_sum = URM.dot(item_pop)

    # 2. Numero di item visti (Profilo Lunghezza)
    user_profile_len = np.ediff1d(sps.csr_matrix(URM).indptr)
    # Evitiamo divisioni per zero
    user_profile_len[user_profile_len == 0] = 1 

    # 3. Media Popolarità nel Profilo
    user_avg_pop = user_pop_sum / user_profile_len

    # 4. Std Popolarità nel Profilo: sqrt( E[X^2] - (E[X])^2 )
    # E[X^2]
    user_pop_sq_sum = URM.dot(item_pop ** 2)
    user_avg_pop_sq = user_pop_sq_sum / user_profile_len

    # Std Dev
    user_std_pop = np.sqrt(np.maximum(0, user_avg_pop_sq - user_avg_pop**2))

    # Aggiunta al DataFrame
    # Creiamo un dataframe temporaneo per fare il merge
    user_features = pd.DataFrame({
        'UserID': np.arange(n_users),
        'User_Avg_Item_Popularity': user_avg_pop,
        'User_Std_Item_Popularity': user_std_pop
    })

    training_dataframe = pd.merge(training_dataframe, user_features, on='UserID', how='left')

    # Feature: Item Popularity dell'item candidato (se non c'è già)
    if 'item_popularity' not in training_dataframe.columns:
        training_dataframe['item_popularity'] = item_pop[training_dataframe['ItemID'].values.astype(int)]
        
    # Feature: Distanza tra la popolarità dell'item candidato e la media dell'utente
    # L'utente sta guardando qualcosa di molto più popolare o di nicchia rispetto al suo solito?
    training_dataframe['Pop_Diff_Item_UserAvg'] = training_dataframe['item_popularity'] - training_dataframe['User_Avg_Item_Popularity']

    # ------------- Embeddings ScaledPureSVD  -------------
    print('--------------- ScaledPureSVD Embeddings ---------------')
    
    # Verifica che il modello sia presente
    if 'ScaledPureSVD' in other_algorithms:
        svd_model = other_algorithms['ScaledPureSVD']
        
        # Recupera le matrici dei fattori
        # Nota: ScaledPureSVD solitamente ha USER_factors e ITEM_factors come IALS
        # Se non li trovi, potrebbero chiamarsi U e V (o Vt)
        u_factors_svd = svd_model.USER_factors
        i_factors_svd = svd_model.ITEM_factors
        
        # Numero di dimensioni latenti da usare (mantienilo basso, es. 5)
        n_latent_svd = 5
        
        # Generazione nomi colonne
        u_cols_svd = [f"SVD_User_Latent_{i}" for i in range(n_latent_svd)]
        i_cols_svd = [f"SVD_Item_Latent_{i}" for i in range(n_latent_svd)]
        
        # --- User Factors ---
        # Creiamo un DataFrame temporaneo per il merge
        user_factors_svd_df = pd.DataFrame(u_factors_svd[:, :n_latent_svd], columns=u_cols_svd)
        user_factors_svd_df['UserID'] = np.arange(n_users)
        
        # Merge con il training dataframe
        training_dataframe = pd.merge(training_dataframe, user_factors_svd_df, on='UserID', how='left')
        
        # --- Item Factors ---
        item_factors_svd_df = pd.DataFrame(i_factors_svd[:, :n_latent_svd], columns=i_cols_svd)
        item_factors_svd_df['ItemID'] = np.arange(n_items)
        
        training_dataframe = pd.merge(training_dataframe, item_factors_svd_df, on='ItemID', how='left')
        
        # Pulizia
        del user_factors_svd_df, item_factors_svd_df
        gc.collect()
        
    else:
        print("ScaledPureSVD model not found in other_algorithms.")


    # -------------- another new one with gemini --------------
    print('--------------- Score Ratios & Rank Differences ---------------')
    
    # LOGICA: Confrontare un modello a Fattori Latenti (Embedding) con uno a Neighborhood (Grafo/Simil)
    # Evitiamo di confrontare RP3beta con P3alpha (troppo simili)
    
    model_pairs = [
        ('IALS', 'SLIMElastic'),      # Deep MF vs Sparse Linear
        ('IALS', 'RP3beta'),          # Deep MF vs Graph
        ('ScaledPureSVD', 'SLIMElastic'), # Linear MF vs Sparse Linear
        ('ScaledPureSVD', 'RP3beta'),     # Linear MF vs Graph
        ('IALS', 'ItemKNNCF'),        # Deep MF vs Pure Item Similarity
        ('SLIMElastic', 'ItemKNNCF'), 
        #EASE, KNN?
    ]

    eps = 1e-6  # evita divisioni per zero

    for algo_a, algo_b in model_pairs:
        # Verifica che entrambi i modelli esistano nel dataframe
        col_score_a = f"{algo_a}_Score"
        col_score_b = f"{algo_b}_Score"
        
        if col_score_a in training_dataframe.columns and col_score_b in training_dataframe.columns:
            
            # --- A. Score Ratio ---
            # Se Ratio > 1, il modello A è più confidente del B
            training_dataframe[f"Ratio_{algo_a}_{algo_b}"] = (training_dataframe[col_score_a] + eps) / (training_dataframe[col_score_b] + eps)

            # --- B. Rank Difference ---
            # Se Positivo: B ha messo l'item più in alto in classifica rispetto ad A
            training_dataframe[f"Diff_{algo_a}_{algo_b}"] = training_dataframe[f"{algo_a}_RankPosition"] - training_dataframe[f"{algo_b}_RankPosition"]
            
        else:
            print(f"Skipping pair {algo_a}-{algo_b}: one or both columns missing.")

    gc.collect()



















    print('training dataframe creato con successo')
    return  training_dataframe





# --------------------------------------------------------------------------------
