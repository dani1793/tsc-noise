#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 00:38:12 2020

@author: daniyalusmani1
"""
from __future__ import print_function
import os
import torch.utils.data as data
import pandas as pd
import numpy as np
import torch


'''class CropTscBalanced2015(data.Dataset):
    def __init__(self, archiveRoot, datasetName, labelIndex, idIndex, transform=None):
        self.samples = []
        self.labels = []
        self.ids = []
        print(os.listdir(archiveRoot))
        dataset = [i for i in os.listdir(archiveRoot) if i == datasetName]
        print(dataset)
        if dataset:
            print('dataset is found')
            data = pd.read_csv(archiveRoot + '/' + dataset[0], header=None, index_col=None)
            print(data.shape)
            print(data.head(5))
            self.labels = torch.Tensor(data.values[:, labelIndex]).long()
            self.ids = torch.Tensor(data.values[:, idIndex]).long()
            self.targets = self.labels
            self.samples = data.drop(columns=[labelIndex, idIndex]).to_numpy()
            print(self.samples.shape)
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
        x = torch.Tensor(x)∏
        return x, y
        '''

class CropTscBalanced2015NoisyVal(data.Dataset):
    def __init__(self, archiveRoot, datasetName, labelIndex, idIndex, iteration = 0, datasetType= "TRAIN", transform=None):
        self.samples = []
        self.labels = []
        self.ids = []
        print(os.listdir(archiveRoot))
        dataset = [i for i in os.listdir(archiveRoot) if i == datasetName]
        print(dataset)
        dataset_path = archiveRoot + '/' + dataset[0]
        if dataset:
            print('dataset is found')
            if iteration == 0:
                dataset_path = archiveRoot + '/' + dataset[0] + '/val_noise_dataset'
                print('fetching data from path %s'%(dataset_path))
                data = pd.read_csv(dataset_path + '/' + dataset[0], header=None, index_col=None)
            else:
                dataset_path = archiveRoot + '/' + dataset[0] + '/val_noise_dataset/iter-' + str(iteration)
                print('fetching data from path %s'%(dataset_path))
                data = pd.read_csv(dataset_path + '/' + dataset[0] + '_' + datasetType + '.csv', header=None, index_col=None)
            print(data.shape)
            print(data.head(5))
            self.labels = torch.Tensor(data.values[:, labelIndex]).long()
            self.ids = torch.Tensor(data.values[:, idIndex]).long()
            self.targets = self.labels
            self.samples = data.drop(columns=[labelIndex, idIndex]).to_numpy()
            print(self.samples.shape)
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

class CropTscBalanced2015(data.Dataset):
    def __init__(self, archiveRoot, datasetName, labelIndex, idIndex, iteration = 0, datasetType= "TRAIN", transform=None):
        self.samples = []
        self.labels = []
        self.ids = []
        print(os.listdir(archiveRoot))
        iterationName = datasetName
        if iteration != 0:
            iterationName = 'iter-%s'%(str(iteration))
        dataset = [i for i in os.listdir(archiveRoot) if i == iterationName]
        print(dataset)
        if dataset:
            print('dataset is found')
            if iteration == 0:
                data = pd.read_csv(archiveRoot + '/' + dataset[0], header=None, index_col=None)
            else:
                data = pd.read_csv(archiveRoot + '/' + dataset[0] + '/' + datasetName + '_' + datasetType + '.csv', header=None, index_col=None)
            print(data.shape)
            print(data.head(5))
            self.labels = torch.Tensor(data.values[:, labelIndex]).long()
            self.ids = torch.Tensor(data.values[:, idIndex]).long()
            self.targets = self.labels
            self.samples = data.drop(columns=[labelIndex, idIndex]).to_numpy()
            print(self.samples.shape)
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