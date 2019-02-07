# -*- coding: utf-8 -*-

from sklearn.neighbors import NearestNeighbors

from gokinjo.backend_base import BackendTransformerBase


class ScikitTransformer(BackendTransformerBase):

    def build_model(self, X, k, c):
        model = NearestNeighbors(n_neighbors=k)
        model.fit(X)
        return model

    def measure_distances(self, X, model, k, c):
        distances, _ = model.kneighbors(X)
        return distances
