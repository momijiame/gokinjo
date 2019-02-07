# -*- coding: utf-8 -*-

__version__ = '0.1.0'

try:
    __GOKINJO_SETUP__
except NameError:
    __GOKINJO_SETUP__ = False

if not __GOKINJO_SETUP__:
    # NOTE: module not found error occurs if imported in setup process
    from .api import knn_extract, knn_kfold_extract  # noqa: F401
