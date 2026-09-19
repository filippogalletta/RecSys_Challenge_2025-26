#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gorgonzola Racing Team - RecSys Challenge 2025/26
Custom Model: FeatureCombinedImplicitALSRecommender

Wrapper su Implicit Alternating Least Squares (iALS) integrato con il framework del PoliMi.
"""

import numpy as np

try:
    import implicit
except ImportError:
    implicit = None

try:
    from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
except ImportError:
    try:
        from RecSys_Course_AT_PoliMi.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
    except ImportError:
        BaseMatrixFactorizationRecommender = object


class FeatureCombinedImplicitALSRecommender(BaseMatrixFactorizationRecommender):
    """
    FeatureCombinedImplicitALSRecommender: wrapper per implicit.als.AlternatingLeastSquares.
    """

    RECOMMENDER_NAME = "FeatureCombinedImplicitALSRecommender"

    def __init__(self, URM_train, verbose=False):
        super().__init__(URM_train, verbose=verbose)
        self.rec = None

    def fit(
        self,
        iterations=15,
        factors=100,
        alpha=1,
        regularization=0.01,
        use_native=True,
        use_cg=True,
        use_gpu=False,
        calculate_training_loss=False,
        num_threads=0,
        **kwargs
    ):
        if implicit is None:
            raise ImportError("La libreria 'implicit' non è installata. Esegui: pip install implicit")

        self.rec = implicit.als.AlternatingLeastSquares(
            factors=factors, 
            regularization=regularization, 
            use_native=use_native, 
            use_cg=use_cg, 
            use_gpu=use_gpu,
            iterations=iterations,
            calculate_training_loss=calculate_training_loss,
            num_threads=num_threads
        )
        
        matrix_to_fit = self.URM_train * alpha
        self.rec.fit(matrix_to_fit, show_progress=self.verbose)

        # Estrazione fattori latenti compatibile sia con CPU che con GPU/array
        user_factors = self.rec.user_factors
        item_factors = self.rec.item_factors
        
        if hasattr(user_factors, "to_numpy"):
            self.USER_factors = user_factors.to_numpy()
            self.ITEM_factors = item_factors.to_numpy()
        else:
            self.USER_factors = np.array(user_factors)
            self.ITEM_factors = np.array(item_factors)
