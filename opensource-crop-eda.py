#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 23:08:13 2020

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

def generateCorrMatrix(title, filename, dataframe):
    fig, ax = plt.subplots()  
    dataset_mean_by_class = _getAverageTimeSeries(dataframe);
    x_df = pd.DataFrame(dataset_mean_by_class).T
    print(x_df.head())
    print(x_df.shape)
    corr = x_df.corr()
    sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
    ax.set_title(title)
    plt.savefig('opensouce-crop-eda-plots/%s.jpg'%(filename),dpi = 1080, format='jpeg')

def averageTimeSeriesCombined(title, filename, dataframe):
    print(dataframe.describe())
    dataset_mean_by_class = _getAverageTimeSeries(dataframe);
    
    fig, ax = plt.subplots()
    
    classColumn, dataColumns = (0, [1,46])
    plt.xlim(0, dataColumns[1] + 8)
    for c in range(dataset_mean_by_class.shape[0]):
        color = plt.cm.hsv(c / dataset_mean_by_class.shape[0])
        plt.plot(dataset_mean_by_class[c], label = 'class %s'%(c+1), color=color)
    plt.title('Time series average')
    plt.xlabel('timesteps')
    plt.ylabel('averaged value')
    plt.legend(loc='upper right', fontsize='x-small')
    plt.savefig('opensouce-crop-eda-plots/%s.jpg'%(filename), dpi = 1080, format='jpeg')

def _getAverageTimeSeries(dataset):
    classColumn, dataColumns =  (0, [1,46])
    dataset_by_class = dataset.groupby(classColumn)
    dataset_mean_by_class = dataset_by_class.mean()
    return dataset_mean_by_class.to_numpy(); 
    

simpleTrain = pd.read_csv('dac-noise/data/UCRArchive_2018/Crop/Crop_TRAIN.tsv', header=None, index_col=None, sep="\t")
simpleVal = pd.read_csv('dac-noise/data/UCRArchive_2018/Crop/Crop_VAL.tsv', header=None, index_col=None, sep="\t")
simpleAllCropDF = pd.concat([simpleTrain, simpleVal], axis=0)
#averageTimeSeriesCombined("UCR Crop Timeseries", "ucr-crop-timeseries", simpleAllCropDF)

#del simpleAllCropDF[0] # remove the labels
generateCorrMatrix("Opensource Crop Dataset Correlation Heatmap", "opensource-crop-dataset-correlation-heatmap", simpleAllCropDF)
