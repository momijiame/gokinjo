#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import importlib

import numpy as np
from sklearn.model_selection import StratifiedKFold

from gokinjo import backend_sklearn
from gokinjo import preprocessing

# default backends
BACKENDS = {
    'sklearn': backend_sklearn.ScikitTransformer,
}

LOG = logging.getLogger(__name__)

# load optional backends
try:
    import annoy  # noqa
except ImportError:
    # annoy is not installed
    LOG.debug('annoy is NOT installed so disable backend')
else:
    # annoy is installed
    LOG.debug('annoy is installed so enable backend')
    backend_annoy = importlib.import_module('gokinjo.backend_annoy')
    BACKENDS['annoy'] = backend_annoy.AnnoyTransformer


def knn_extract(xtr, ytr, xte, k=1, normalize=None,
                backend=None, backend_params=None):
    """k-NN feature extraction

    this function is typically used to generate k-NN feature for test data.
    or when you generate k-NN feature for training data with your own CV.

    :param xtr: training data for extraction (feature vector)
    :param ytr: training data for extraction (target label)
    :param xte: target data for extraction (feature vector)
    :param k: number of neighbors
    :param normalize: normalize scaling if you want (choice from ['standard', 'minmax'])
    :param backend: k-NN implementation (default: sklearn)
    :param backend_params: backend parameters if needed
    :return: extracted feature vector
    """

    # default backend
    if backend is None:
        backend = 'sklearn'

    # lookup transformer
    transformer_class = BACKENDS.get(backend)
    if transformer_class is None:
        raise ValueError('choice from {backends}'.format(backends=','.join(BACKENDS.keys())))

    # common parameters
    transformer_params = {
        'n_neighbors': k,
    }

    # update parameters for backend specified
    if backend_params is None:
        backend_params = {}
    transformer_params.update(backend_params)

    # preprocessing
    if normalize is not None:
        xtr, xte = preprocessing.normalize(normalize, xtr, xte)

    # fit and transform
    transformer = transformer_class(**transformer_params)
    transformer.fit(xtr, ytr)
    return transformer.transform(xte)


def knn_kfold_extract(xtr, ytr, k=1, normalize=None,
                      folds=5, shuffle=True, random_state=None,
                      backend=None, backend_params=None):
    """k-NN feature extraction with k-Fold

    this function is typically used to generate k-NN feature for training data.
    to prevent over-fitting by using k-Fold when generating.

    :param xtr: training data for extraction (feature vector)
    :param ytr: training data for extraction (target label)
    :param k: number of neighbors
    :param normalize: normalize scaling if you want (choice from ['standard', 'minmax'])
    :param folds: a number of fold or scikit-learn KFold object
    :param shuffle: is data shuffled when cv
    :param random_state: random seed value
    :param backend: k-NN implementation (default: sklearn)
    :param backend_params: backend parameters if needed
    :return: extracted feature vector
    """
    classes = np.unique(ytr)
    feature_with_index = np.empty([0, 1 + len(classes) * k])

    if isinstance(folds, int):
        kf = StratifiedKFold(n_splits=folds,
                             shuffle=shuffle,
                             random_state=random_state)
    else:
        # customized CV
        kf = folds

    for train_index, test_index in kf.split(xtr, ytr):
        X_train, X_test = xtr[train_index], xtr[test_index]
        y_train, y_test = ytr[train_index], ytr[test_index]  # noqa: F841

        # feature extraction
        extracted_feature = knn_extract(X_train, y_train, X_test, k, normalize,
                                        backend, backend_params)
        # concat index
        reshaped_index = test_index.reshape((len(test_index), 1))
        extracted_feature_with_index = np.concatenate((reshaped_index,
                                                       extracted_feature),
                                                      axis=1)

        # store working space
        feature_with_index = np.append(feature_with_index,
                                       extracted_feature_with_index,
                                       axis=0)

    # sort by index
    sorted_feature = feature_with_index[feature_with_index[:, 0].argsort(), :]
    # return without index
    return sorted_feature[:, 1:]
