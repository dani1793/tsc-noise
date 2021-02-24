#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 00:36:19 2020

@author: daniyalusmani1
"""


import pandas as pd
import argparse

import numpy as np
import matplotlib.pyplot as plt
import os


parser = argparse.ArgumentParser(description='Plot creation for TSC Average time series plot',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset_path', default=None, type=str, help='path to data file')
parser.add_argument('--title', default=None, type=str, help='Title of plot')
parser.add_argument('--dataset_name', default=None, type=str, help='name of dataset')


global labelCol


def getConfigForDataset():
    if(args.dataset_name == "UCR_Crop"):
        return (0, [1,46])
    elif(args.dataset_name == "ai_crop"):
        return (9,[0,7])

#TODO: have to set the label column and data columns manually as the datasets are not consistent
args = parser.parse_args()
def generateAverageTimeSeriesPlot():
    # for crop dataset dataset = pd.read_csv(args.dataset_path, sep='\t', header=None)
    dataset = pd.read_csv(args.dataset_path, header=None) # for ai_crop
    del dataset[8] # for ai_crop
    print(dataset.describe())
    print(dataset.head())

    dataset_mean_by_class = getAverageTimeSeries(dataset);
    classColumn, dataColumns = getConfigForDataset()

    fig, ax = plt.subplots()
    
    # plt.xlim(0, dataColumns[1] + 8) for crop
    plt.xlim(0, dataColumns[1] + 2) # for ai_crop
    for c in range(dataset_mean_by_class.shape[0]):
        plt.plot(dataset_mean_by_class[c], label = 'class %s'%(c+1))
    plt.title('Time series average')
    plt.xlabel('timesteps')
    plt.ylabel('averaged value')
    plt.legend(loc='upper right', fontsize='x-small')
    plt.savefig('dac-noise/results/plots/time-series-average-%s.jpg'%(args.dataset_name), dpi = 1080, format='jpeg')

def getAverageTimeSeries(dataset):
    classColumn, dataColumns = getConfigForDataset()
    dataset_by_class = dataset.groupby(classColumn)
    dataset_mean_by_class = dataset_by_class.mean()
    return dataset_mean_by_class.to_numpy(); 


generateAverageTimeSeriesPlot()

#python3 tsc-avg-time-series-plot.py --dataset_path /m/cs/scratch/ai_croppro/daniyal/dac-noise --title "Time series average" --dataset_name ai_crop