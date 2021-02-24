#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 21:40:28 2020

@author: daniyalusmani1
"""


import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils as ut
import config as cfg
import torch
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def generateCorrMatrix(title, filename):
    fig, ax = plt.subplots(figsize=(20,10))  
    x_df = pd.DataFrame(X)
    corr = x_df.corr()
    sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
    ax.set_title(title, fontsize=14)
    plt.savefig('%s.png'%(filename))
    

def undersampleBalance(X, y, ids):
    concat = np.concatenate([X, np.expand_dims(ids, axis=1), np.expand_dims(y, axis=1)], axis= 1)
    print(concat[0:5])
    x_df = pd.DataFrame(concat)
    print(x_df.head())
    shuffled_df = x_df.sample(frac=1,random_state=4)

    # Put all the one class in a separate dataset.
    one_df = shuffled_df.loc[shuffled_df[13] == 1]

    #Randomly select 6576 observations from zero (majority class)
    zero_df = shuffled_df.loc[shuffled_df[13] == 0].sample(n=6576,random_state=42)

    # Concatenate both dataframes again
    normalized_df = pd.concat([zero_df, one_df])
    #print(normalized_df.shape)
    #print(normalized_df.iloc[:, 0:12].shape)
    #print(normalized_df.iloc[:, 12].shape)
    #print(normalized_df.iloc[:, 13].shape)
    
    #print(normalized_df.iloc[:, 0:12].head())
    #print(normalized_df.iloc[:, 12].head())
    #print(normalized_df.iloc[:, 13].head())
    
    return normalized_df.iloc[:, 0:12].to_numpy(), normalized_df.iloc[:, 13].to_numpy(), normalized_df.iloc[:, 12].to_numpy() 


def handleMissingData(data):
    x_df = pd.DataFrame(data)
    x_df = x_df.T.interpolate().bfill().T
    return x_df.to_numpy()

def generateBalancedNonMissingDataWithImputation(X, y, ids):
    X, y, ids = undersampleBalance(X, y, ids)
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    x_non_missing = imputer.fit_transform(X)
    concat = np.concatenate([x_non_missing, np.expand_dims(ids, axis=1), np.expand_dims(y, axis=1)], axis= 1)
    pd.DataFrame(concat).to_csv("crop_tsc_balanced_imputed_2015.csv", header=False, index=False)
    
    
def generateBalancedNonMissingData(X, y, ids):
    x_non_missing = handleMissingData(X)
    X, y, ids = undersampleBalance(x_non_missing, y, ids)
    concat = np.concatenate([X, np.expand_dims(ids, axis=1), np.expand_dims(y, axis=1)], axis= 1)
    pd.DataFrame(concat).to_csv("crop_tsc_balanced_filled_2015.csv", header=False, index=False)

def generateBalancedNonMissingDataWithImputationAndPCA(sourceCSV):
    data = pd.read_csv(sourceCSV, index_col=None, header=None)
    # Principle Component Analysis (PCA)
    var_threshold = 0.98
    pca_obj = PCA(n_components=var_threshold) # Create PCA object
    dataPCA = pca_obj.fit_transform(StandardScaler().fit_transform(data.iloc[:,0:8])) 
    dfDataPCA = pd.DataFrame(dataPCA)
    dfDataPCA.insert(8,8,data.iloc[:,8]) # 8th index is the ids of the samples (these are removed when performing analysis)
    dfDataPCA.insert(9,9,data.iloc[:,9]) # 9th is the labels 
    dfDataPCA.to_csv("crop_tsc_balanced_imputed_PCA_2015.csv", header=False, index=False)

def checkNullValues(X):
    print('checking Null values in tsc features')
    concat = np.concatenate([X, np.expand_dims(ids, axis=1), np.expand_dims(y, axis=1)], axis= 1)
    combined_df = pd.DataFrame(concat)
    print(combined_df.isnull().sum())


X, y, ids = ut.load_ndvi_uts(cfg.data_path, ut.as_list(2015), cfg.balance_flag)

#generateBalancedNonMissingData(X, y, ids)
#generateBalancedNonMissingDataWithImputation(X, y, ids)
#generateBalancedNonMissingDataWithImputationAndPCA('crop_tsc_balanced_imputed_2015.csv')

# To generate Correlation matrix of balanced and unbalanced
#generateCorrMatrix('Imbalanced Correlation Matrix', 'imbalanced-correlation')
#X, y, ids = undersampleBalance(X, y, ids)
#print(X.shape)
#print(y.shape)
#print(ids.shape)

#generateCorrMatrix('Balanced Correlation Matrix', 'balanced-correlation')


# unique_elements, counts_elements = np.unique(y, return_counts=True)
# print("Frequency of unique labels:")
# print(np.asarray((unique_elements, counts_elements)))

# mdl = LogisticRegression(random_state=cfg.random_state)
# imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# scaler = preprocessing.StandardScaler()

# clf = Pipeline([("imputer", imputer), ("scalar", scaler), ("mdl", mdl)])
# auc_cv = cross_val_score(clf, X, y, cv=cfg.cv, scoring=cfg.scoring)
# print(np.mean(auc_cv))


#data = pd.read_csv('crop_tsc_balanced_imputed_2015.csv', header=None, index_col=None)
#print(data.head(5))
#labels = torch.Tensor(data.values[:, 9]).long()
#print(labels.unique())

data = pd.read_csv('crop_tsc_balanced_imputed_PCA_2015.csv', header=None, index_col=None)
print(data.head(5))
labels = torch.Tensor(data.values[:, 9]).long()
print(labels.unique())




