#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import numpy as np
from sklearn.model_selection import KFold

from gokinjo import knn_extract
from gokinjo import knn_kfold_extract


def test_knn_extract_k1():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 0, 1])
    target = np.array([[-1], [0], [2], [3]])

    extracted = knn_extract(X, y, target)

    # k=1, label=0
    assert all(extracted[:, 0] == np.array([2, 1, 0, 1]))
    # k=1, label=1
    assert all(extracted[:, 1] == np.array([4, 3, 1, 0]))


def test_knn_extract_k2():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 1, 1])
    target = np.array([[-1], [0], [2], [3]])

    extracted = knn_extract(X, y, target, k=2)

    # k=1, label=0
    assert all(extracted[:, 0] == np.array([2, 1, 0, 0]))
    # k=2, label=0
    assert all(extracted[:, 1] == np.array([2 + 3, 1 + 2, 0 + 1, 0 + 1]))
    # k=1, label=1
    assert all(extracted[:, 2] == np.array([5, 4, 2, 1]))
    # k=2, label=1
    assert all(extracted[:, 3] == np.array([5 + 6, 4 + 5, 2 + 3, 1 + 2]))


def test_knn_extract_normalize_minmax():
    X = np.array([[1], [2], [3]], dtype=float)  # => [0, 0.5, 1]
    y = np.array([0, 0, 1])
    target = np.array([[-1], [0], [2], [3]], dtype=float)  # => [-1, -0.5, 0.5, 1]

    extracted = knn_extract(X, y, target, normalize='minmax')

    # k=1, label=0
    assert all(extracted[:, 0] == np.array([1.0, 0.5, 0.0, 0.5]))
    # k=1, label=1
    assert all(extracted[:, 1] == np.array([2.0, 1.5, 0.5, 0.0]))


def test_knn_kfold_extract():
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([0, 0, 0, 1, 1, 1])

    extracted = knn_kfold_extract(X, y, folds=3, shuffle=False)

    # split: [1, 4], [2, 4], [3, 6]
    # k=1, label=0
    assert all(extracted[:, 0] == np.array([1, 1, 1, 1, 2, 4]))
    # k=1, label=1
    assert all(extracted[:, 1] == np.array([4, 2, 1, 1, 1, 1]))


def test_knn_kfold_extract_custom_cv():
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([0, 0, 0, 1, 1, 1])

    loo_cv = KFold(n_splits=6, shuffle=False)
    extracted = knn_kfold_extract(X, y, folds=loo_cv)

    # k=1, label=0
    assert all(extracted[:, 0] == np.array([1, 1, 1, 1, 2, 3]))
    # k=1, label=1
    assert all(extracted[:, 1] == np.array([3, 2, 1, 1, 1, 1]))


def test_knn_extract_backend_annoy():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 0, 1])
    target = np.array([[-1], [0], [2], [3]])

    extracted = knn_extract(X, y, target, backend='annoy')

    # k=1, label=0
    assert all(extracted[:, 0] == np.array([2, 1, 0, 1]))
    # k=1, label=1
    assert all(extracted[:, 1] == np.array([4, 3, 1, 0]))


if __name__ == '__main__':
    pytest.main(['-v', __file__])
