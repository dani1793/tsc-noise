import pandas as pd
import numpy as np
import collections
import argparse
import os

import utils as ut
import config as cfg

from sklearn.impute import SimpleImputer
from sklearn.utils import resample

import numpy as np
from sklearn.model_selection import StratifiedKFold

def saveFile(data, iteration, filename):
    folderToSaveIn = "dac-noise/data/ai_crop/val_noise_dataset"
    scriptDir = os.path.dirname(os.path.realpath('__file__'))
    absFilePath = os.path.join(scriptDir, folderToSaveIn)
    fileToSave = absFilePath + "/iter-%s/%s.csv"%(str(iteration), filename)
    if iteration == 0:
        fileToSave = absFilePath + "/%s.csv"%(filename)
    print(fileToSave)
    pd.DataFrame(data).to_csv(fileToSave, header=False, index=False)

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

LABEL_COL = 9 # for ai_crop

print("--- performing imputation ---")
imputedDataset = performImputation()

labels = imputedDataset[:,LABEL_COL]
data = imputedDataset[:,:-1]


print("--- Finding majority class ---")
majorityClass, minorityClass, _, _1 = findMajorityClass(labels)
print("majority class ::", majorityClass)
print("minority class :: ", minorityClass)

print("--- Balancing dataset ---")
balancedDataset = balanceOutDataset(imputedDataset, LABEL_COL, majorityClass, minorityClass)

balancedData = np.array(balancedDataset.to_numpy()[:,:-1])
balancedLabels = np.array(balancedDataset.to_numpy()[:,LABEL_COL])

print("Testing balance of imputed dataset")
balancedMajorityClass, balancedMinorityClass, balancedMajorityClassSample, balancedMinorityClassSample = findMajorityClass(balancedLabels)
print("do both classes have same sample size :: ", balancedMajorityClassSample == balancedMinorityClassSample)
assert balancedMajorityClassSample == balancedMinorityClassSample, "Balanced Classes do not have same size"

# save the balanced dataset for checking afterwards
saveFile(balancedDataset.to_numpy(), 0, "ai_crop_balanced")

skf = StratifiedKFold(n_splits=5) # check what does this do??
print(skf.get_n_splits(balancedData, balancedLabels))
print(skf)
iteration = 1;


for train_index, test_index in skf.split(balancedData, balancedLabels):
    print("TRAIN:", train_index, "TEST:", test_index)
    validationDataset = balancedDataset.to_numpy()[test_index,:]
    trainingDataset = balancedDataset.to_numpy()[train_index,:]
    

    print("Testing balance of training dataset")
    balancedMajorityClass, balancedMinorityClass, balancedMajorityClassSample, balancedMinorityClassSample = findMajorityClass(trainingDataset[:,LABEL_COL])
    print("do both classes have same sample size :: ", balancedMajorityClassSample == balancedMinorityClassSample)
    

    print("Testing balance of validation dataset")
    balancedMajorityClass, balancedMinorityClass, balancedMajorityClassSample, balancedMinorityClassSample = findMajorityClass(validationDataset[:,LABEL_COL])
    print("do both classes have same sample size :: ", balancedMajorityClassSample == balancedMinorityClassSample)

    saveFile(test_index, iteration, "VAL_indexes")
    saveFile(train_index, iteration, "TRAIN_indexes")
    
    saveFile(np.array(validationDataset), iteration, "ai_crop_VAL")
    saveFile(np.array(trainingDataset), iteration, "ai_crop_TRAIN")
    
    iteration = iteration + 1

