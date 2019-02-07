#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

SCALERS = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
}


def normalize(scaler, xtr, xte):
    # look up scaler
    scaler_class = SCALERS.get(scaler)
    if scaler_class is None:
        raise ValueError('choice from {scalers}'.format(scalers=','.join(SCALERS.keys())))

    # fit to training data
    transformer = scaler_class()
    transformer.fit(xtr)

    # transform
    return transformer.transform(xtr), transformer.transform(xte)
