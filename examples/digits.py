#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.manifold import TSNE

from gokinjo import knn_kfold_extract

"""usage example of gokinjo with Digits dataset"""


def main():
    # load dataset
    dataset = datasets.load_digits()
    X = dataset.data
    y = dataset.target

    # classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # k-fold parameter
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # pattern 1: raw data
    score = cross_validate(clf, X, y, cv=skf)
    print('mean accuracy (raw):', score['test_score'].mean())

    # pattern 2: raw -> k-NN transform
    X_knn_feature_raw = knn_kfold_extract(X, y)
    score = cross_validate(clf, X_knn_feature_raw, y, cv=skf)
    print('mean accuracy (raw -> k-NN):', score['test_score'].mean())

    # pattern 3: raw -> t-SNE transform
    tsne = TSNE()
    X_transformed = tsne.fit_transform(X)
    score = cross_validate(clf, X_transformed, y, cv=skf)
    print('mean accuracy (raw -> t-SNE):', score['test_score'].mean())

    # pattern 4: raw -> t-SNE -> k-NN stacking
    X_knn_feature_tsne = knn_kfold_extract(X_transformed, y)
    score = cross_validate(clf, X_knn_feature_tsne, y, cv=skf)
    print('mean accuracy (raw -> t-SNE -> k-NN):', score['test_score'].mean())


if __name__ == '__main__':
    main()
