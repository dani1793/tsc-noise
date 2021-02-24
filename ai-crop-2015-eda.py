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
import config as cfg
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('crop_tsc_balanced_imputed_2015.csv', index_col=None, header=None)
# Principle Component Analysis (PCA)
var_threshold = 0.98
pca_obj = PCA(n_components=var_threshold) # Create PCA object
data_Transformed = pca_obj.fit_transform(StandardScaler().fit_transform(data.iloc[:,0:8])) 
df_data_Transformed = pd.DataFrame(data_Transformed)
df_data_Transformed.insert(8,8,data.iloc[:,9])
print(df_data_Transformed.head(5))
groups = df_data_Transformed.groupby(8)
print(groups.head(5));




'''

for name, group in groups:
    plt.plot(group.iloc[:,0], group.iloc[:,1], marker="o", linestyle="", label=name)
plt.legend()
plt.title("Scatter plot for PCA")
plt.savefig('pca-scatter.png')


X, y, ids = ut.load_ndvi_uts(cfg.data_path, ut.as_list(2015), cfg.balance_flag)
print(X[0:5])
orig = pd.read_csv('ndvi_1_30_365_all_years_for_daniyal.csv') 
print(orig.head(5))

data = pd.read_csv('crop_tsc_balanced_imputed_2015.csv', header=None, index_col=None)
data = data.drop(columns=[9])
print(data.head(5))
#labels = torch.Tensor(data.values[:, 9]).long()
#print(labels.unique())

# Feature 8 is selected because of unusually high values
data = data.to_numpy()
sel = VarianceThreshold(threshold=.8)
print(sel.fit_transform(data))

data = pd.read_csv('crop_tsc_balanced_imputed_2015.csv', header=None, index_col=None)
data = data.drop(columns=[8,9])
print(data.head(5))

data = data.to_numpy()
sel = VarianceThreshold(threshold=.05)
print(sel.fit_transform(data))


data = pd.read_csv('crop_tsc_balanced_imputed_2015.csv', index_col=None, 
                   names=["Feature 0", "Feature 1", "Feature 2",
                              "Feature 3", "Feature 4", "Feature 5",
                              "Feature 6", "Feature 7", "Feature 8",
                              "label"])
data = data.drop(columns=["Feature 8","label"])
print(data.head(5))
print(data.iloc[:,np.hstack(([0],range(1,8)))])

for i in range(0,8):
    for j in range(i, 8):
        data.plot(kind='scatter', x='Feature %s'%(str(i)),y='Feature %s'%(str(j)))
        plt.savefig('pair-plot-Feature-%s-Feature%s.png'%(str(i), str(j)))    


data = pd.read_csv('crop_tsc_balanced_imputed_2015.csv', index_col=None, 
                   names=["Feature 0", "Feature 1", "Feature 2",
                              "Feature 3", "Feature 4", "Feature 5",
                              "Feature 6", "Feature 7", "Feature 8",
                              "label"])
data = data.drop(columns=["Feature 8","label"])
# Principle Component Analysis (PCA)
var_threshold = 0.98
pca_obj = PCA(n_components=var_threshold) # Create PCA object
data_Transformed = pca_obj.fit_transform(StandardScaler().fit_transform(data)) # Transform the initial features
data_Transformed = pd.DataFrame(data_Transformed, columns=["Feature 0", "Feature 1", "Feature 2",
                              "Feature 3", "Feature 4", "Feature 5",
                              "Feature 6", "Feature 7"])
#fig, ax = plt.subplots(figsize=(20,10))  
#corr = data_Transformed.corr()

#sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
#ax.set_title("Correlation after PCA", fontsize=14)
#plt.savefig('pca-correlation.png')
print(data_Transformed.head(5))

fig, ax = plt.subplots(figsize=(20,10))  
data_Transformed.boxplot(column=["Feature 0", "Feature 1", "Feature 2",
                              "Feature 3", "Feature 4", "Feature 5",
                              "Feature 6", "Feature 7"])
plt.savefig('pca-box-wisker.png')
'''






