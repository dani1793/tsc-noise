# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import torch.utils.data as data
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import torch


def getSimpleUCRArchive(archiveRoot, datasetName, noise = 0, transform=None):
    dataset = [i for i in os.listdir(archiveRoot) if i == datasetName]
    if dataset: 
        print("dataset is found")
        if noise == 0:
            df_train = pd.read_csv(archiveRoot + '/' + dataset[0] + '/' + dataset[0] + '_' + 'TRAIN' + '.tsv', sep='\t', header=None)
            df_test = pd.read_csv(archiveRoot + '/' + dataset[0] + '/' + dataset[0] + '_' + 'VAL' + '.tsv', sep='\t', header=None)
        else:
            df_train = pd.read_csv(archiveRoot + '/' + dataset[0] + '/' + dataset[0] + '_' + str(noise) + '_' + 'TRAIN' + '.tsv', sep='\t', header=None)
            df_test = pd.read_csv(archiveRoot + '/' + dataset[0] + '/' + dataset[0] + '_' + str(noise) + '_' + 'VAL' + '.tsv', sep='\t', header=None)
            
        y_train = df_train.values[:, 0] - 1
        y_test = df_test.values[:, 0] - 1
    
        x_train = df_train.drop(columns=[0])
        x_test = df_test.drop(columns=[0])
    
        x_train.columns = range(x_train.shape[1])
        x_test.columns = range(x_test.shape[1])
    
        x_train = x_train.values
        x_test = x_test.values
    
        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_
    
        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
        
        return (x_train.copy(), y_train.copy(), x_test.copy(),
                                   y_test.copy())
        

    else:
        raise FileNotFoundError    

class UCRArchive(data.Dataset):
    
    def __init__(self, archiveRoot, datasetName, iteration = 0, datasetType = 'TRAIN', noise = 0, transform=None):
        self.samples = []
        self.labels = []
        dataset = [i for i in os.listdir(archiveRoot) if i == datasetName]
        if dataset:
            print('dataset is found')
            print('fetching data for UCR Archive datatype %s for noise %s'%(datasetType, noise))
            dataset_path = archiveRoot + '/' + dataset[0]
            if iteration != 0:
                dataset_path = archiveRoot + '/' + dataset[0] + '/iter-' + str(iteration)
            print('fetching data from path %s'%(dataset_path))
            if noise == 0:
                data = pd.read_csv(dataset_path + '/' + dataset[0] + '_' + datasetType + '.tsv', sep='\t', header=None)
            else:
                data = pd.read_csv(dataset_path + '/' + dataset[0] + '_' + str(noise) + '_' + datasetType + '.tsv', sep='\t', header=None)
            self.labels = torch.Tensor(data.values[:, 0] - 1).long()
            self.targets = self.labels
            self.samples = data.drop(columns=[0]).to_numpy()
            self.data = self.samples
            
            std_ = self.samples.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0            
            self.samples = (self.samples - self.samples.mean(axis=1, keepdims=True)) / std_
            
        else:
            raise FileNotFoundError;

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        x =  np.expand_dims(x, axis=0)
        y = self.labels[idx]
        x = torch.Tensor(x)
        return x, y

