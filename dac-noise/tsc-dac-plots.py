#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:26:04 2020

@author: daniyalusmani1
"""

import pandas as pd
import argparse

import numpy as np
import matplotlib.pyplot as plt
import os


parser = argparse.ArgumentParser(description='Plot creation for TSC DAC architecture',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--orig_dataset', default=None, type=str, help='original dataset which in which noise was introduced')
parser.add_argument('--noisy_dataset', default=None, type=str, help='noisy dataset which in which noise was introduced')
parser.add_argument('--noisy_percentage', default=0, type=float, help='noisy percentage')
parser.add_argument('--dataset', default=None, type=str, help='Name of dataset')
parser.add_argument('--exp_name', default=None, type=str, help='Name of experiment')
parser.add_argument('--kfold', default=None, type=int, help='Number of kfold experiments')

args = parser.parse_args()
def generateOriginalDatasetClassDist():
    
    orig = pd.read_csv(args.orig_dataset, sep='\t', header=None)
    origLabels = orig[0] - 1
    classes = len(np.unique(origLabels)) 
    fig, ax = plt.subplots()
    
    n, bins, patches = ax.hist(origLabels, density=1, edgecolor='black', bins=classes, linewidth=1.2)
    plt.xlabel('Classes')
    plt.ylabel('count')
    plt.title('Class distribution in original %s set'%(args.dataset))
    
    plt.savefig('results/plots/%s-original-class-dist.png'%(args.dataset))
    plt.close()

def generateNoisyDatasetClassDist():
    rootPath = args.noisy_dataset + '_' + str(args.noisy_percentage)
    noisyTrain = pd.read_csv(rootPath +  '_TRAIN.tsv', sep='\t', header=None)
    noisyVal = pd.read_csv(rootPath + '_VAL.tsv', sep='\t', header=None)
    noisyLabels = np.append(noisyTrain[0], noisyVal[0])
    classes = len(np.unique(noisyLabels)) 
    fig, ax = plt.subplots()
    
    n, bins, patches = ax.hist(noisyLabels, density=1, edgecolor='black', bins=classes, linewidth=1.2)
    plt.xlabel('Classes')
    plt.ylabel('count')
    plt.title('Class distribution in noisy %s set'%(args.dataset))
    
    plt.savefig('results/plots/%s-noisy-%s-class-dist.png'%(args.dataset, str(args.noisy_percentage)))
    plt.close()
    
def _extractAccumuatedActivations(path, files):
    activations = []
    for file in files:
        epoch = np.load(path + '/' + file,  allow_pickle=True);
        if len(activations) == 0:
            activations = epoch.mean(axis=0);
            activations = np.expand_dims(activations, axis = 1)
        else:
            activations = np.append(activations, np.expand_dims(epoch.mean(axis=0), axis = 1), axis = 1)
        # print(activations.shape)
    return activations

def _plotAccumulatedActivations(activations, activationType):
    for c in range(activations.shape[0]):
        if c == activations.shape[0] - 1:
            plt.plot(activations[c], label = '%s'%('Abstention'))
        else:
            plt.plot(activations[c], label = 'class %s'%(str(c)))
    if args.noisy_percentage:
        plt.title('%s Activation energies for %s with noise %s'%(activationType, args.dataset, str(args.noisy_percentage)))
    else:
        plt.title('%s Activation energies for %s'%(activationType, args.dataset))
    plt.xlabel('Epochs')
    plt.ylabel('mean activation energy / epoch')
    plt.legend(loc='upper right', fontsize='xx-small')
    if args.noisy_percentage:
       plt.savefig('results/plots/%s-%s-%s-%s-activations.jpg'%(args.dataset, args.exp_name, activationType, str(args.noisy_percentage)), dpi = 1080, format='jpeg') 
    else:
        plt.savefig('results/plots/%s-%s-%s-activations.jpg'%(args.dataset, args.exp_name, activationType), dpi = 1080, format='jpeg')
        
    plt.close()

def _plotLossFunctionPerEpoch(trainFile, valFile):
    train = np.load(trainFile,  allow_pickle=True)
    val = np.load(valFile,  allow_pickle=True)
    
    plt.plot(train, label = '%s'%('training loss'))
    plt.plot(val, label = '%s'%('validation loss'))
    if args.noisy_percentage:
        plt.title('Loss plot for %s with noise %s'%(args.dataset, str(args.noisy_percentage)))
    else:
        plt.title('Loss plot for %s'%(args.dataset))
   
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right', fontsize='xx-small')
    if args.noisy_percentage:
       plt.savefig('results/plots/%s-%s-%s-loss.jpg'%(args.dataset, args.exp_name, str(args.noisy_percentage)), dpi = 1080, format='jpeg') 
    else:
        plt.savefig('results/plots/%s-%s-loss.jpg'%(args.dataset, args.exp_name), dpi = 1080, format='jpeg')
        
    plt.close()

def _plotF1ScoreWithoutAbstensionPerEpoch(trainFile, valFile):
    train = np.load(trainFile,  allow_pickle=True)
    val = np.load(valFile,  allow_pickle=True)
    
    plt.plot(train, label = '%s'%('training f1score'))
    plt.plot(val, label = '%s'%('validation f1score'))
    if args.noisy_percentage:
        plt.title('F1 Score plot for %s with noise %s'%(args.dataset, str(args.noisy_percentage)))
    else:
        plt.title('F1 Score plot for %s'%(args.dataset))
   
    plt.xlabel('Epochs')
    plt.ylabel('F1score')
    plt.legend(loc='upper right', fontsize='xx-small')
    if args.noisy_percentage:
       plt.savefig('results/plots/%s-%s-%s-f1score-without-abstained.jpg'%(args.dataset, args.exp_name, str(args.noisy_percentage)), dpi = 1080, format='jpeg') 
    else:
        plt.savefig('results/plots/%s-%s-f1score-without-abstained.jpg'%(args.dataset, args.exp_name), dpi = 1080, format='jpeg')
        
    plt.close()
    
    
def _plotF1ScorePerEpoch(trainFile, valFile):
    train = np.load(trainFile,  allow_pickle=True)
    val = np.load(valFile,  allow_pickle=True)
    
    plt.plot(train, label = '%s'%('training f1score'))
    plt.plot(val, label = '%s'%('validation f1score'))
    if args.noisy_percentage:
        plt.title('F1 Score plot for %s with noise %s'%(args.dataset, str(args.noisy_percentage)))
    else:
        plt.title('F1 Score plot for %s'%(args.dataset))
   
    plt.xlabel('Epochs')
    plt.ylabel('F1score')
    plt.legend(loc='upper right', fontsize='xx-small')
    if args.noisy_percentage:
       plt.savefig('results/plots/%s-%s-%s-f1score.jpg'%(args.dataset, args.exp_name, str(args.noisy_percentage)), dpi = 1080, format='jpeg') 
    else:
        plt.savefig('results/plots/%s-%s-f1score.jpg'%(args.dataset, args.exp_name), dpi = 1080, format='jpeg')
        
    plt.close()   
    
def _plotLossFunctionAndF1ScorePerEpoch(trainLossFile, trainF1scoreFile, valLossFile, valF1scoreFile):
    trainLoss = np.load(trainLossFile,  allow_pickle=True)
    trainF1score = np.load(trainF1scoreFile,  allow_pickle=True)
    
    valLoss = np.load(valLossFile,  allow_pickle=True)
    valF1score = np.load(valF1scoreFile,  allow_pickle=True)
    
    plt.plot(trainLoss, label = '%s'%('training loss'))
    plt.plot(valLoss, label = '%s'%('validation loss'))
    
    plt.plot(trainF1score, label = '%s'%('training F1 score'))
    plt.plot(valF1score, label = '%s'%('validation F1 score'))
    
    if args.noisy_percentage:
        plt.title('Loss plot for %s with noise %s'%(args.dataset, str(args.noisy_percentage)))
    else:
        plt.title('Loss plot for %s'%(args.dataset))
   
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right', fontsize='xx-small')
    if args.noisy_percentage:
       plt.savefig('results/plots/%s-%s-%s-loss-f1score.jpg'%(args.dataset, args.exp_name, str(args.noisy_percentage)), dpi = 1080, format='jpeg') 
    else:
        plt.savefig('results/plots/%s-%s-loss-f1score.jpg'%(args.dataset, args.exp_name), dpi = 1080, format='jpeg')
        
    plt.close()
    
    
    
def _plotLossFunctionPerEpochKfold(losses):
    ind = 1
    for (trainLoss, valLoss) in losses:
        plt.plot(trainLoss, label = 'training loss for kfold %s'%(ind))
        plt.plot(valLoss, label = 'validation loss for kfold %s'%(ind))
        ind = ind + 1
        
    
    if args.noisy_percentage:
        plt.title('Loss plot for %s with noise %s'%(args.dataset, str(args.noisy_percentage)))
    else:
        plt.title('Loss plot for %s'%(args.dataset))
    

def _plotAccVsAbstainedPerEpoch(trainAbstainedFile, trainAccFile, valAbstainedFile, valAccFile):
    trainAbstained = np.load(trainAbstainedFile,  allow_pickle=True)
    valAbstained = np.load(valAbstainedFile,  allow_pickle=True)
    
    trainAcc = np.load(trainAccFile, allow_pickle=True)
    valAcc = np.load(valAccFile, allow_pickle=True)
    
    plt.plot(trainAbstained, label = '%s'%('training abstained'))
    plt.plot(valAbstained, label = '%s'%('Validation abstained'))

    plt.plot(trainAcc, label = '%s'%('training Accuracy'))
    plt.plot(valAcc, label = '%s'%('Validation Accuracy'))
    
    if args.noisy_percentage:
        plt.title('Abstaiened vs Accuracy plot for %s with noise %s'%(args.dataset, str(args.noisy_percentage)))
    else:
        plt.title('Abstaiened vs Accuracy plot for %s'%(args.dataset))
   
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right', fontsize='xx-small')
    if args.noisy_percentage:
       plt.savefig('results/plots/%s-%s-%s-acc-vs-abstained.jpg'%(args.dataset, args.exp_name, str(args.noisy_percentage)), dpi = 1080, format='jpeg') 
    else:
        plt.savefig('results/plots/%s-%s-acc-vs-abstained.jpg'%(args.dataset, args.exp_name), dpi = 1080, format='jpeg')
        
    plt.close()
    
    

def generateActivationEnergyPlotsPerEpoch():
    rootPath = 'results/' + args.dataset.lower() + '/' + str(args.noisy_percentage)
    if args.exp_name:
        rootPath = 'results/' + args.dataset.lower() + '/' + str(args.exp_name)

    script_dir = os.path.dirname(os.path.realpath('__file__'))
   # script_dir = os.path.dirname(__file__) # <-- absolute dir the script is in
    abs_file_path = os.path.join(script_dir, rootPath)
    print(abs_file_path)
    entries = os.listdir(abs_file_path)
    entries = np.array(entries)
    trainIndices = ['train' in entry for entry in entries]
    valIndices = ['val' in entry for entry in entries]
    valActivations = entries[valIndices]
    trainActivations = entries[trainIndices]
    
    accumlatedTrainActivations = _extractAccumuatedActivations(abs_file_path, trainActivations)
    accumlatedValActivations = _extractAccumuatedActivations(abs_file_path, valActivations)
    
    _plotAccumulatedActivations(accumlatedTrainActivations, 'train')
    _plotAccumulatedActivations(accumlatedValActivations, 'val')
    
    
def _getLossesForKFold(absPath):
    trainLosses = []
    valLosses = []
    for i in range(args.kfold):
        trainLossFileName = absPath + '/' + args.exp_name + '.train-loss.npy'
        valLossFileName = absPath + '/' + args.exp_name + '.val-loss.npy'
        trainLosses.append(np.load(trainLossFileName, allow_pickle=True))
        valLosses.append(np.load(valLossFileName, allow_pickle=True))

    return trainLosses, valLosses
      
def generateLossPerEpoch():
    rootPath = 'results/' + args.dataset.lower() + '/' + str(args.exp_name) + '/loss'
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    abs_file_path = os.path.join(script_dir, rootPath)
    trainLoss = abs_file_path + '/' + args.exp_name + '.train-loss.npy'
    valLoss = abs_file_path + '/' + args.exp_name + '.val-loss.npy'

    _plotLossFunctionPerEpoch(trainLoss, valLoss)

def generateF1ScorePerEpoch():
    rootPath = 'results/' + args.dataset.lower() + '/' + str(args.exp_name) + '/f1score'
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    abs_file_path = os.path.join(script_dir, rootPath)
    trainF1Score = abs_file_path + '/' + args.exp_name + '.train-f1score.npy'
    valF1Score = abs_file_path + '/' + args.exp_name + '.val-f1score.npy'

    _plotF1ScorePerEpoch(trainF1Score, valF1Score)

def generateF1ScoreWithoutAbstensionPerEpoch():
    rootPath = 'results/' + args.dataset.lower() + '/' + str(args.exp_name) + '/f1score'
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    abs_file_path = os.path.join(script_dir, rootPath)
    trainF1Score = abs_file_path + '/' + args.exp_name + '.train-f1score-without-abstained.npy'
    valF1Score = abs_file_path + '/' + args.exp_name + '.val-f1score-without-abstained.npy'

    _plotF1ScoreWithoutAbstensionPerEpoch(trainF1Score, valF1Score)
    


def generateLossAndF1ScorePerEpoch():
    rootPathLoss = 'results/' + args.dataset.lower() + '/' + str(args.exp_name) + '/loss'
    rootPathF1score = 'results/' + args.dataset.lower() + '/' + str(args.exp_name) + '/f1score'
    
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    abs_file_path_loss = os.path.join(script_dir, rootPathLoss)
    abs_file_path_f1score = os.path.join(script_dir, rootPathF1score)
    
    trainLoss = abs_file_path_loss + '/' + args.exp_name + '.train-loss.npy'
    valLoss = abs_file_path_loss + '/' + args.exp_name + '.val-loss.npy'
    
    trainF1Score = abs_file_path_f1score + '/' + args.exp_name + '.train-f1score.npy'
    valF1Score = abs_file_path_f1score + '/' + args.exp_name + '.val-f1score.npy'
    
    _plotLossFunctionAndF1ScorePerEpoch(trainLoss, trainF1Score, valLoss, valF1Score)
    
    

def generateLossPerEpochKfold():
    rootPath = 'results/' + args.dataset.lower() + '/' + str(args.exp_name) + '/loss'
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    abs_file_path = os.path.join(script_dir, rootPath)
    trainingLosses, valLosses = _getLossesForKFold(abs_file_path)
    
    _plotLossFunctionPerEpochKfold((trainingLosses, valLosses))
    
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right', fontsize='xx-small')
    if args.noisy_percentage:
       plt.savefig('results/plots/%s-%s-%s-loss.jpg'%(args.dataset, args.exp_name, str(args.noisy_percentage)), dpi = 1080, format='jpeg') 
    else:
        plt.savefig('results/plots/%s-%s-loss.jpg'%(args.dataset, args.exp_name), dpi = 1080, format='jpeg')
        
    plt.close()
    


def generateAbstensionVsAccPerEpoch():
    rootPathAbstained = 'results/' + args.dataset.lower() + '/' + str(args.exp_name) + '/abstained'
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    abs_file_path = os.path.join(script_dir, rootPathAbstained)
    trainAbstained = abs_file_path + '/' + args.exp_name + '.train-abstained.npy'
    valAbstained = abs_file_path + '/' + args.exp_name + '.val-abstained.npy'

    rootPathAcc = 'results/' + args.dataset.lower() + '/' + str(args.exp_name) + '/acc'
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    abs_file_path = os.path.join(script_dir, rootPathAcc)
    trainAcc = abs_file_path + '/' + args.exp_name + '.train-acc.npy'
    valAcc = abs_file_path + '/' + args.exp_name + '.val-acc.npy'
    
    _plotAccVsAbstainedPerEpoch(trainAbstained, trainAcc, valAbstained, valAcc)


def getf1Score():
    rootPathAbstained = 'results/' + args.dataset.lower() + '/' + str(args.exp_name) + '/f1score'
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    abs_file_path = os.path.join(script_dir, rootPathAbstained)
    trainf1ScoreFileName = abs_file_path + '/' + args.exp_name + '.train-f1score.npy'
    valf1ScoreFileName = abs_file_path + '/' + args.exp_name + '.val-f1score.npy'
    
    trainF1Score = np.load(trainf1ScoreFileName ,  allow_pickle=True)
    valF1Score = np.load(valf1ScoreFileName,  allow_pickle=True)
    print(np.mean(trainF1Score))
    print(np.mean(valF1Score))
    


#generateOriginalDatasetClassDist()
#generateNoisyDatasetClassDist()


if args.kfold:
    generateLossPerEpochKfold()

else:
    #generateActivationEnergyPlotsPerEpoch()
    generateLossPerEpoch()
    generateF1ScorePerEpoch()
    generateF1ScoreWithoutAbstensionPerEpoch()
    
    #generateLossAndF1ScorePerEpoch()
    generateAbstensionVsAccPerEpoch()
    #getf1Score()



#generateLossPlotPerEpoch()
    

