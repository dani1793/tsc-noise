#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hiremas1
"""

import pandas as pd
import numpy as np
import config as cfg
from sklearn.model_selection import StratifiedShuffleSplit


np.random.seed(5)


def load_ndvi_uts(filepath: str, years, balance_flag: int):
    """
    load NDVI univariate time series from csv file
    """
    df = pd.read_csv(filepath, dtype={7: str})
    t = df.columns[~df.columns.isin(cfg.attribute_vars)]

    df = df.loc[df.year.isin(years)].reset_index(drop=True)

    # if balance flag is 1 then sequentially load data per year and balance it
    # this is like stratified sampling across years
    if balance_flag == 1:
        for i, year in enumerate(years):
            df_year = df.loc[df.year == year].reset_index(drop=True)
            x = df_year.loc[:, t].values
            y = df_year.loc[:, "loss"].values
            z = df_year.loc[:, "new_ID"].values
            if i == 0:
                X, y, ids = balance_data(x, y, z)
            else:
                X_year, y_year, ids_year = balance_data(x, y, z)
                X = np.vstack([X, X_year])
                y = np.hstack([y, y_year])
                ids = np.hstack([ids, ids_year])
    else:  # bulk load
        X = df.loc[:, t].values
        y = df.loss.values
        ids = df["new_ID"].values

    X[X == cfg.fill_value] = np.nan
    X, y, ids = shuffle(X, y, ids)
    X, y = X.astype(np.float32), y.astype(np.int64)
    return X, y, ids


def balance_data(X, y, ids):
    """
    Number of crop loss is always smaller than no loss.
    Randomly select no-loss according to number of loss fields
    """
    idx1 = y == 1
    idx0 = y == 0

    x1, y1, ids1 = X[idx1], y[idx1], ids[idx1]
    x0, y0, ids0 = X[idx0], y[idx0], ids[idx0]

    n1 = np.sum(y == 1)
    n0 = np.sum(y == 0)

    idx0 = np.arange(n0)
    idx0 = np.random.permutation(idx0)
    x0, y0, ids0 = x0[idx0[:n1]], y0[idx0[:n1]], ids0[idx0[:n1]]

    X = np.vstack((x0, x1))
    y = np.hstack((y0, y1))
    ids = np.hstack((ids0, ids1))
    return X, y, ids


def split_3fold(X, y, trp=0.6, vap=0.8):
    """
    Split data into train, test and validation set
    """
    assert len(X) == len(y), print("Error: X and y length mismatch.")

    nimages = len(X)
    np.random.seed(cfg.random_state)
    idx = np.random.permutation(nimages)
    idx_tr = idx[: int(trp * nimages)]
    idx_va = idx[int(trp * nimages) : int(vap * nimages)]
    idx_te = idx[int(vap * nimages) :]

    X_tr = X[idx_tr]
    y_tr = y[idx_tr]

    X_te = X[idx_te]
    y_te = y[idx_te]

    X_va = X[idx_va]
    y_va = y[idx_va]
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def split_2fold(X, y, ids, trp=0.8):
    """
    Split data into 2 folds
    """
    assert len(X) == len(y), print("Error: X and y length mismatch.")

    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=trp, random_state=cfg.random_state
    )

    for idx_tr, idx_te in sss.split(X, y):
        X_tr, y_tr, ids_tr = X[idx_tr], y[idx_tr], ids[idx_tr]
        X_te, y_te, ids_te = X[idx_te], y[idx_te], ids[idx_te]

    return X_tr, y_tr, ids_tr, X_te, y_te, ids_te

def permute_like(arr, ref_arr):
    """
    return the indices of elements of arr that are in the same order as
    elements in ref_arr
    """
    index = []
    for (i, val) in enumerate(ref_arr):
        index.append(np.where(arr == val)[0][0])
    return index


def shuffle(X, y, ids):
    idx = np.random.permutation(np.arange(len(X)))
    X = X[idx]
    y = y[idx]
    ids = ids[idx]
    return X, y, ids

def missing_data_rate(im_seq_list):
    total_seq_len = 0
    total_zeros = 0
    for im_seq in im_seq_list:
        total_seq_len += len(im_seq)
        binary_qa_mask = im_seq[-1]
        total_zeros += np.sum(np.sum(binary_qa_mask, axis=(1, 2)) == 0)
    return total_zeros / total_seq_len


def match_columns(a, b):
    asum = np.nansum(a, axis=0)
    bsum = np.nansum(b, axis=0)
    keep = (asum != 0) & (bsum != 0)
    aout = a.T[keep]
    bout = b.T[keep]
    return aout.T, bout.T


def as_list(x):
    if type(x) is list:
        return x
    else:
        return [x]

