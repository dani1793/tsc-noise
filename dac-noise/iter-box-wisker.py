#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:10:50 2020

@author: daniyalusmani1
"""

# python3 iter-box-wisker.py --dataset crop --exp_name 2-lstm-crop-noise-0.3-epoch-5000-dac-learning-epoch-40-lr-0.001 --noise_percentage 0.3
# python3 iter-box-wisker.py --dataset crop --exp_name 2-lstm-crop-noise-0.5-epoch-5000-dac-learning-epoch-100-lr-0.001 --noise_percentage 0.5
# python3 iter-box-wisker.py --dataset crop --exp_name 2-lstm-crop-noise-0.75-epoch-5000-dac-learning-epoch-400-lr-0.001 --noise_percentage 0.75

import argparse
import os

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Box wisker plot for iterations', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--exp_name', default=None, type=str, help='experiment name without iteration.')
parser.add_argument('--dataset', default=None, type=str, help='name of dataset')
parser.add_argument('--noise_percentage', default=0, type=float, help='percentage of noise')
args = parser.parse_args()


def loadIterationFile(iteration):
    
    iter_exp_name = str(args.exp_name) + '-iter' +str(iteration)
    rootPath = 'results/' + args.dataset.lower() + '/' + iter_exp_name + '/f1score'
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    abs_file_path = os.path.join(script_dir, rootPath)
    valF1ScoreFile = abs_file_path + '/' + iter_exp_name + '.val-f1score.npy'
    print("loading file with path %s"%(valF1ScoreFile))
    score = np.load(valF1ScoreFile,  allow_pickle=True)
    return score

def savePlot(boxplot):
    rootPath = 'results/plots/'
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    abs_file_path = os.path.join(script_dir, rootPath)
    boxplotSavePath = abs_file_path + '/' + str(args.exp_name)+'-f1score-compare.jpg'
    plt.title('LSTM validation F1score comparison for \n %s dataset with %s %s noise '%(args.dataset, str(args.noise_percentage * 100), "%" ))
    plt.xlabel('Iterations')
    plt.ylabel('F1 Score')
    plt.savefig(boxplotSavePath, dpi = 1080, format='jpeg')

finalScore = []
for i in range(5):
    score = loadIterationFile(i+1)
    
    finalScore.append(score)

print(np.array(finalScore).T.shape)
df = pd.DataFrame(np.array(finalScore).T,columns=['1', '2', '3', '4', '5'])
boxplot = df.boxplot(showfliers=False)
savePlot(boxplot)

