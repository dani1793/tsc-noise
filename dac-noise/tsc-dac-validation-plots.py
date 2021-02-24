#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 21:19:12 2021

@author: daniyalusmani1
"""
import pandas as pd
import argparse

import numpy as np
import matplotlib.pyplot as plt
import os


dataset = "ai_crop" # "crop"
exp2lstmBase = [
    "2-lstm-crop-noise-0.3-epoch-5000-dac-learning-epoch-40-lr-0.001-iter2", 
    "2-lstm-crop-noise-0.5-epoch-5000-dac-learning-epoch-100-lr-0.001-iter2",
    "2-lstm-crop-noise-0.75-epoch-5000-dac-learning-epoch-400-lr-0.001-iter3"
    ]

exp2inceptionBase = [
    "inception-simple-crop-noise-0.3-epoch-300-dac-learning-epoch-50-lr-0.1-iter2",
    "inception-simple-crop-noise-0.5-epoch-500-dac-learning-epoch-150-lr-0.1-iter2",
    "inception-simple-crop-noise-0.75-epoch-500-dac-learning-epoch-150-lr-0.1-iter3"
    ]

exp2lstmBaseWithoutAbstension = [
    "2-lstm-crop-noise-0.3-epoch-5000-dac-learning-epoch-40-lr-0.001-without-abstained-iter2",
    "2-lstm-crop-noise-0.5-epoch-4500-dac-learning-epoch-100-lr-0.001-without-abstained-iter2",
    "2-lstm-crop-noise-0.75-epoch-5000-dac-learning-epoch-400-lr-0.001-without-abstained-iter3"
    ]

exp3lstmBase = [
    "2-lstm-ai_crop-epoch-5000-dac-learning-epoch-250-lr-0.1-pow_0.2-240_1-iter3",
    ]




exp3inception = [
    "inception-simple-ai_crop-epoch-800-dac-learning-epoch-10-lr-0.0001-iter3",
    ]


def getValLossDataForExp(experiment):
    rootPath = 'results/' + dataset + '/' + experiment + '/loss'
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    abs_file_path = os.path.join(script_dir, rootPath)
    valLoss = abs_file_path + '/' + experiment + '.val-loss.npy'
    return valLoss
    
def getValF1DataForExp(experiment):
    rootPath = 'results/' + dataset + '/' + experiment + '/f1score'
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    abs_file_path = os.path.join(script_dir, rootPath)
    valF1Score = abs_file_path + '/' + experiment + '.val-f1score.npy'
    return valF1Score

def getValF1WithoutAbstainedDataForExp(experiment):
    rootPath = 'results/' + dataset + '/' + experiment + '/f1score'
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    abs_file_path = os.path.join(script_dir, rootPath)
    valF1Score = abs_file_path + '/' + experiment + '.val-f1score-without-abstained.npy'
    return valF1Score

def getTrainF1WithoutAbstainedDataForExp(experiment):
    rootPath = 'results/' + dataset + '/' + experiment + '/f1score'
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    abs_file_path = os.path.join(script_dir, rootPath)
    valF1Score = abs_file_path + '/' + experiment + '.train-f1score-without-abstained.npy'
    return valF1Score

def getMovingAverageOfSeries(series, window):
    numbers_series = pd.Series(series)
    windows = numbers_series.rolling(window)
    moving_averages = windows.mean()

    moving_averages_list = moving_averages.tolist()
    without_nans = moving_averages_list[window - 1:]
    return without_nans
    

def generatePlot(experiments, plotType, architectureType, fileName):
    windowSize = 15
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14) 
    plt.rc('axes', labelsize=14)
    for (exp, noiseLevel,color) in experiments:
        file = np.load(exp,  allow_pickle=True)
        if noiseLevel == 0:
            plt.plot(getMovingAverageOfSeries(file, windowSize), color)
        else:
            plt.plot(getMovingAverageOfSeries(file, windowSize), color, label = '%s'%(str(noiseLevel)))
            plt.legend(loc='upper right', fontsize='x-small')
        #plt.plot(file, color, label = '%s'%(str(noiseLevel)))

    plt.xlabel('Epochs')
    plt.ylabel(plotType)

    plt.savefig('results/plots/experiment-3-ma15-epoch-800-%s-%s.jpg'%(architectureType, fileName), dpi = 1080, format='jpeg')
    plt.close()

#generatePlot(zip([getValLossDataForExp(exp) for exp in exp2lstmBase], [0.3,0.5,0.75], ['g-','r--','b-.']), 'Loss', "lstm", 'loss')
#generatePlot(zip([getValF1DataForExp(exp) for exp in exp2lstmBase], [0.3,0.5,0.75], ['g-','r--','b-.']), 'F1 score', "lstm", 'f1score')
#generatePlot(zip([getValF1WithoutAbstainedDataForExp(exp) for exp in exp2lstmBaseWithoutAbstension], [0.3,0.5,0.75], ['g-','r--','b-.']), 'F1 score pruned', "lstm", "f1score-without-abstained")
#generatePlot(zip([getTrainF1WithoutAbstainedDataForExp(exp) for exp in exp2lstmBaseWithoutAbstension], [0.3,0.5,0.75], ['g-','r--','b-.']), 'F1 score pruned', "lstm", 'train-f1score-without-abstained')

#generatePlot(zip([getValLossDataForExp(exp) for exp in exp2inceptionBase], [0.3,0.5,0.75], ['g-','r--','b-.']), 'Loss', "inception", 'loss')
#generatePlot(zip([getValF1DataForExp(exp) for exp in exp2inceptionBase], [0.3,0.5,0.75], ['g-','r--','b-.']), 'F1 score', "inception", 'f1score')
#generatePlot(zip([getValF1WithoutAbstainedDataForExp(exp) for exp in exp2inceptionBase], [0.3,0.5,0.75], ['g-','r--','b-.']), 'F1 score Pruned', "inception", 'f1score-without-abstained')
#generatePlot(zip([getTrainF1WithoutAbstainedDataForExp(exp) for exp in exp2inceptionBase], [0.3,0.5,0.75], ['g-','r--','b-.']), 'F1 score pruned', "inception", 'train-f1score-without-abstained')


#generatePlot(zip([getValLossDataForExp(exp3lstmBase[0])], [0], ['g-','r--','b-.']), 'Loss', "lstm", 'loss')
#generatePlot(zip([getValF1DataForExp(exp3lstmBase[0])], [0], ['g-','r--','b-.']), 'F1 score', "lstm", 'f1score')
#generatePlot(zip([getValF1WithoutAbstainedDataForExp(exp3lstmBase[0]),getTrainF1WithoutAbstainedDataForExp(exp3lstmBase[0])], ["F1score Val", "F1score Train"], ['g-','b-.']), 'F1 score pruned', "lstm", "train-val-f1score-without-abstained")
#generatePlot(zip([getTrainF1WithoutAbstainedDataForExp(exp) for exp in exp3inception], [], ['g-','r--','b-.']), 'F1 score pruned', "inception", 'train-f1score-without-abstained')



generatePlot(zip([getValLossDataForExp(exp3inception[0])], [0], ['g-','r--','b-.']), 'Loss', "inception", 'loss')
generatePlot(zip([getValF1DataForExp(exp3inception[0])], [0], ['g-','r--','b-.']), 'F1 score', "inception", 'f1score')
generatePlot(zip([getValF1WithoutAbstainedDataForExp(exp3inception[0]),getTrainF1WithoutAbstainedDataForExp(exp3inception[0])], ["F1score Val", "F1score Train"], ['g-','b-.']), 'F1 score pruned', "inception", "train-val-f1score-without-abstained")
#generatePlot(zip([getTrainF1WithoutAbstainedDataForExp(exp) for exp in exp3inception], [], ['g-','r--','b-.']), 'F1 score pruned', "inception", 'train-f1score-without-abstained')



