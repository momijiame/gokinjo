# -*- coding: utf-8 -*-

import numpy as np
from annoy import AnnoyIndex

from gokinjo.backend_base import BackendTransformerBase


class AnnoyTransformer(BackendTransformerBase):

    def __init__(self, n_trees=10, search_k=-1, n_neighbors=1):
        super().__init__(n_neighbors)
        # backend specified parameters
        self.n_trees_ = n_trees
        self.search_k_ = search_k

    def build_model(self, X, k, c):
        model = AnnoyIndex(X.shape[1], metric='euclidean')
        for j, x in enumerate(X):
            model.add_item(j, x)
        model.build(n_trees=self.n_trees_)
        return model

    def measure_distances(self, X, model, k, c):
        distances = np.array([self._neighbor_distance(model, x, k) for x in X])
        return distances

    def _neighbor_distance(self, model, x, k):
        _, distances = model.get_nns_by_vector(x, k,
                                               search_k=self.search_k_,
                                               include_distances=True)
        return distances

    def get_params(self, deep=True):
        params = super().get_params(deep)
        annoy_params = {
            'n_trees': self.n_trees_,
            'search_k': self.search_k_,
        }
        params.update(annoy_params)
        return params
