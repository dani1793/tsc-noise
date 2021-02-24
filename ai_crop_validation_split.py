#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:04:22 2020

@author: daniyalusmani1

TODO:
    1. Load the ai_crop dataset
    2. get indices of 0's
    3. select 25% of dataset
    4. balance the remaining classes
    5. create 5 iterations
    
    To regenerate the samples we need the validation indices and training indices, 
    these validation indices and training indices could be used to regenerate the samples
"""



import pandas as pd
import numpy as np
import collections
import argparse
import os

import utils as ut
import config as cfg

from sklearn.impute import SimpleImputer
from sklearn.utils import resample

parser = argparse.ArgumentParser(description='Generate validation split for ai crop dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', default=None, type=str, help='Name of dataset')
parser.add_argument('--filename', default=None, type=str, help='File name to load the data from')
parser.add_argument('--save_folder', default=None, type=str, help='folder to save the generated dataset in')
parser.add_argument('--iter', default=None, type=str, help='iteration to generate')

args = parser.parse_args()

def getDataset():
    filePath = 'data/' + args.dataset.lower() + args.filename.lower()
    scriptDir = os.path.dirname(os.path.realpath('__file__'))
    absFilePath = os.path.join(scriptDir, filePath)
    dataset = np.load(absFilePath, allow_pickle=True)
    return dataset


def getIndicesFromMask(mask):
    labelIndices = np.where(mask)[0]
    return labelIndices

def extractIndicesForLabel(dataset, label):
    labelMask = dataset == label
    return getIndicesFromMask(labelMask)

def extractValidationIndices(indices, percentage = 0.25):
    return np.random.choice(indices, int(len(indices)*percentage))

def getNotSelectedIndices(dataset, selectedIndices):
    notSelectedIndicesMask = [ i not in selectedIndices for i in range(len(dataset))]
    return getIndicesFromMask(notSelectedIndicesMask)

def balanceOutDataset(dataset, labelCol, majorityClass, minorityClass):
    df = pd.DataFrame(dataset);
    dfMinority = df[df[labelCol] == minorityClass]
    dfMajority = df[df[labelCol] == majorityClass]
    
    dfMajorityDownsampled = resample(dfMajority, 
                                 replace=False,    # sample without replacement
                                 n_samples=dfMinority.shape[0],     # to match minority class
                                 random_state=123) # reproducible results
    
    dfDownsampled = pd.concat([dfMajorityDownsampled, dfMinority])
    print("Class distribution for downsampled dataset")
    print(dfDownsampled.describe())
    
    return dfDownsampled
    
def findMajorityClass(dataset):
    ctr = collections.Counter(dataset)
    print("Frequency of the elements in the List : ",ctr)
    itemList = list(ctr.items())
    print(itemList)
    print(itemList[0][0])
    
    majorityClass = itemList[0][0]
    majorityClassSample = itemList[0][1]
    minorityClass = itemList[-1][0]
    minorityClassSample =itemList[-1][1]
    return majorityClass, minorityClass, majorityClassSample, minorityClassSample

def performImputation():
    X, y, ids = ut.load_ndvi_uts(cfg.data_path, ut.as_list(2015), cfg.balance_flag)
    print("nonimputed Dataset length")
    print(X.shape)
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    x_non_missing = imputer.fit_transform(X)
    concat = np.concatenate([x_non_missing, np.expand_dims(ids, axis=1), np.expand_dims(y, axis=1)], axis= 1)
    print("Imputed dataset length")
    print(concat.shape)
    return concat

def saveFile(data, filename):
    folderToSaveIn = args.save_folder
    scriptDir = os.path.dirname(os.path.realpath('__file__'))
    absFilePath = os.path.join(scriptDir, folderToSaveIn)
    fileToSave = absFilePath + "/%s_%s.csv"%(args.dataset, filename)
    pd.DataFrame(data).to_csv(fileToSave, header=False, index=False)


LABEL_COL = 9 # for ai_crop

print("--- performing imputation ---")
imputedDataset = performImputation()
print(imputedDataset[:,LABEL_COL])

print("--- extracting the labels for imputed dataset ---")
labels = np.array(imputedDataset[:,LABEL_COL].tolist())
print("%s labels extracted"%(len(labels)))

print("--- extract indices for samples with 0 class --- ")
selectedLabel = labels[0] # zero label sample
indicesForSelectedLabel = extractIndicesForLabel(labels, selectedLabel)
print("Total number of samples with 0 class", len(indicesForSelectedLabel))
print("All indices equal to selected label :: ", np.all(labels[indicesForSelectedLabel] == selectedLabel))
assert np.all(labels[indicesForSelectedLabel] == selectedLabel), "All indices are not equal to selected label"

print("--- Extracting validation indices ---")
validationIndices = extractValidationIndices(indicesForSelectedLabel)
print("selected validation indices length :: ", len(validationIndices))

print("--- Training indices ---")
trainingIndices = getNotSelectedIndices(labels, validationIndices)
print("Training indices length :: ", len(trainingIndices))
print("training Indices and Validation Indices are disjoint set :: ", set(trainingIndices).isdisjoint(set(validationIndices)))
assert set(trainingIndices).isdisjoint(set(validationIndices)), "training Indices and Validation Indices are not disjoint set "

print("--- generating dataset from indices selected ---")
trainingSamples = imputedDataset[trainingIndices]
validationSamples = imputedDataset[validationIndices]

print("--- Finding majority class ---")
majorityClass, minorityClass, _, _1 = findMajorityClass(labels[trainingIndices])
print("majority class ::", majorityClass)
print("minority class :: ", minorityClass)

print("--- Balancing dataset ---")
balancedTrainingDataset = balanceOutDataset(trainingSamples, LABEL_COL, majorityClass, minorityClass)
balancedTrainedlabels = np.array(balancedTrainingDataset.to_numpy()[:,LABEL_COL].tolist())

print("--- Class distribution after balanced dataset ---")
balancedMajorityClass, balancedMinorityClass, balancedMajorityClassSample, balancedMinorityClassSample = findMajorityClass(balancedTrainedlabels)
print("do both classes have same sample size :: ", balancedMajorityClassSample == balancedMinorityClassSample)
assert balancedMajorityClassSample == balancedMinorityClassSample, "Balanced Classes do not have same size"


print("--- saving generated data to provided file name ---")
saveFile(validationIndices, "validationIndices")
saveFile(trainingIndices, "trainingIndices")

saveFile(np.array(validationSamples), "VAL")
saveFile(np.array(balancedTrainingDataset), "TRAIN")



    
    

    
    
    
    




