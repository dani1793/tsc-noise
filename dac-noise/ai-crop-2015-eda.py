#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 23:08:56 2020

@author: daniyalusmani1
"""

import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import utils as ut
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline


X, y, ids = ut.load_ndvi_uts(cfg.data_path, ut.as_list(2015), cfg.balance_flag)
print(X.head(5))
orig = pd.read_csv('../ndvi_1_30_365_all_years_for_daniyal.csv') 
print(orig.head(5))

data = pd.read_csv('../crop_tsc_balanced_imputed_2015.csv', header=None, index_col=None)
data = data.drop(columns=[9])
print(data.head(5))
#labels = torch.Tensor(data.values[:, 9]).long()
#print(labels.unique())

data = data.to_numpy()
sel = VarianceThreshold(threshold=.8)
print(sel.fit_transform(data))