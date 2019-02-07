#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from gokinjo import knn_kfold_extract

"""usage example of gokinjo with XOR scatter"""


def main():
    # generate sample data
    x0 = np.random.rand(500) - 0.5
    x1 = np.random.rand(500) - 0.5
    X = np.array(list(zip(x0, x1)))
    y = np.array([1 if i0 * i1 > 0 else 0 for i0, i1 in X])

    # plot original data
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('original data')
    plt.scatter(X[:, 0], X[:, 1], c=y)

    # k-NN feature extraction with k-Fold
    extracted_features = knn_kfold_extract(X, y)

    # plot extracted feature data
    plt.subplot(1, 2, 2)
    plt.title('extracted features')
    plt.scatter(extracted_features[:, 0], extracted_features[:, 1], c=y)

    # common settings
    axes = plt.gcf().get_axes()
    for axe in axes:
        axe.grid()

    plt.show()


if __name__ == '__main__':
    main()
