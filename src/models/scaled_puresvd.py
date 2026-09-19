#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gorgonzola Racing Team - RecSys Challenge 2025/26
Custom Model: ScaledPureSVDRecommender

Variante di PureSVD con riscalatura esponenziale di utenti e item basata sulla popolarità.
"""

import numpy as np
import scipy.sparse as sps

try:
    from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
except ImportError:
    try:
        from RecSys_Course_AT_PoliMi.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
    except ImportError:
        PureSVDRecommender = object


class ScaledPureSVDRecommender(PureSVDRecommender):
    """
    ScaledPureSVDRecommender: applica una trasformazione diagonale alla URM prima della SVD.
    """

    RECOMMENDER_NAME = "ScaledPureSVDRecommender"

    def __init__(self, URM_train, verbose=True):
        super(ScaledPureSVDRecommender, self).__init__(URM_train, verbose=verbose)

    def fit(self, num_factors=100, random_seed=None, scaling_items=1.0, scaling_users=1.0):
        item_pop = np.ediff1d(sps.csc_matrix(self.URM_train).indptr)
        item_scaling_matrix = sps.diags(np.power(item_pop + 1e-6, scaling_items))

        user_pop = np.ediff1d(sps.csr_matrix(self.URM_train).indptr)
        user_scaling_matrix = sps.diags(np.power(user_pop + 1e-6, scaling_users))

        self.URM_train = user_scaling_matrix.dot(self.URM_train).dot(item_scaling_matrix)

        super(ScaledPureSVDRecommender, self).fit(num_factors=num_factors, random_seed=random_seed)
