# -*- coding: utf-8 -*-

import abc
from collections import OrderedDict

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils import check_X_y
from sklearn.utils import check_array


class BackendTransformerBase(TransformerMixin,
                             metaclass=abc.ABCMeta):

    """Base class for k-NN feature extraction implements"""

    def __init__(self, n_neighbors=1):
        # common parameters
        self.n_neighbors_ = n_neighbors
        # internal resources
        self.C_ = None
        self.trained_models_ = OrderedDict()

    @abc.abstractmethod
    def build_model(self, X, k, c):
        raise NotImplementedError()

    def fit(self, X, y=None):
        # validate
        check_X_y(X, y)

        # unique class labels
        self.C_ = np.unique(y)

        # generate models: C * k
        for i, c in enumerate(self.C_):
            # extract training data of class specified
            X_c = X[y == c]

            for k in range(1, self.n_neighbors_ + 1):
                # train and store
                self.trained_models_[(c, k)] = self.build_model(X_c, k, c)

        return self

    @abc.abstractmethod
    def measure_distances(self, X, model, k, c):
        raise NotImplementedError()

    def transform(self, X, copy=None):
        # validate
        check_array(X)

        # prepare to store feature vector
        features = np.zeros([len(X), len(self.C_) * self.n_neighbors_])

        # feature extraction
        for i, ((c, k), model) in enumerate(self.trained_models_.items()):
            distances = self.measure_distances(X, model, k, c)
            sum_distances = np.sum(distances, axis=1)
            features[:, i] = sum_distances

        return features

    def get_params(self, deep=True):
        # common parameters
        return {
            'n_neighbors': self.n_neighbors_,
        }
