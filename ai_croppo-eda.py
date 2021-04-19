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
    plt.savefig('faa-crop-eda-plots/%s.jpg'%(filename),dpi = 1080, format='jpeg')

def averageTimeSeriesCombined(title, filename, dataframe, classLabels):
    print(dataframe.describe())
    dataset_mean_by_class = _getAverageTimeSeries(dataframe);
    
    fig, ax = plt.subplots()
    print(dataset_mean_by_class)
    classColumn, dataColumns = (9, [0,7])
    plt.xlim(0, dataColumns[1] + 1)
    for c in range(dataset_mean_by_class.shape[0]):
        color = plt.cm.hsv(c / dataset_mean_by_class.shape[0])
        print(dataset_mean_by_class[c][dataColumns[0]:dataColumns[1]:])
        plt.plot(dataset_mean_by_class[c][dataColumns[0]:dataColumns[1]:], label = classLabels[c], color=color)
    plt.title('Time series average')
    plt.xlabel('timesteps')
    plt.ylabel('averaged value')
    plt.legend(loc='upper right', fontsize='x-small')
    plt.savefig('faa-crop-eda-plots/%s.jpg'%(filename), dpi = 1080, format='jpeg')

def _getAverageTimeSeries(dataset):
    classColumn, dataColumns =  (9, [0,7])
    dataset_by_class = dataset.groupby(classColumn)
    dataset_mean_by_class = dataset_by_class.mean()
    return dataset_mean_by_class.to_numpy(); 
    

cropDF = pd.read_csv('dac-noise/data/ai_crop/ai_crop/val_noise_dataset/ai_crop.csv', header=None, index_col=None)
#print(cropDF)
averageTimeSeriesCombined("FAA Crop Timeseries", "faa-crop-timeseries", cropDF, ["No Crop Loss", "Crop Loss"])

#del cropDF[9] # remove the labels
del cropDF[8] # remove the labels
generateCorrMatrix("FAA Crop Dataset Correlation Heatmap", "faa-crop-dataset-correlation-heatmap", cropDF)
