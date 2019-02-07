#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from sklearn import datasets

from gokinjo import knn_kfold_extract

"""usage example of gokinjo with Breast Cancer dataset"""


def main():
    # load dataset
    dataset = datasets.load_breast_cancer()
    X = dataset.data
    y = dataset.target

    # k-NN feature extraction with k-Fold
    extracted_features = knn_kfold_extract(X, y)

    # plot extracted feature data
    plt.title('extracted features')
    plt.scatter(extracted_features[:, 0], extracted_features[:, 1], c=y)

    # common settings
    axes = plt.gcf().get_axes()
    for axe in axes:
        axe.grid()

    plt.show()


if __name__ == '__main__':
    main()
