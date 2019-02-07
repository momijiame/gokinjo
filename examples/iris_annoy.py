#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import datasets
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from gokinjo import knn_kfold_extract

"""usage example of gokinjo with annoy backend"""


def main():
    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target

    # use annoy backend
    extracted_features = knn_kfold_extract(X, y, backend='annoy')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(extracted_features[:, 0],
               extracted_features[:, 1],
               extracted_features[:, 2],
               c=y)
    ax.set_title('extracted features (3D)')

    axes = plt.gcf().get_axes()
    for axe in axes:
        axe.grid()

    plt.show()


if __name__ == '__main__':
    main()
