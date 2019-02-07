#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from gokinjo import preprocessing


def test_normalize_standard():
    """z-score normalization"""
    xtr = np.array([[1], [2]], dtype=float)
    xte = np.array([[3], [4]], dtype=float)
    normalized_xtr, normalized_xte = preprocessing.normalize('standard', xtr, xte)

    assert all(normalized_xtr == ((xtr - xtr.mean()) / xtr.std()))
    # NOTE: test data normalized by train data stats
    assert all(normalized_xte == ((xte - xtr.mean()) / xtr.std()))


def test_normalize_minmax():
    """min-max normalization"""
    xtr = np.array([[1], [2]], dtype=float)
    xte = np.array([[3], [4]], dtype=float)
    normalized_xtr, normalized_xte = preprocessing.normalize('minmax', xtr, xte)

    assert all(normalized_xtr == ((xtr - xtr.min()) / (xtr.max() - xtr.min())))
    # NOTE: test data normalized by train data stats
    assert all(normalized_xte == ((xte - xtr.min()) / (xtr.max() - xtr.min())))


if __name__ == '__main__':
    pytest.main(['-v', __file__])
