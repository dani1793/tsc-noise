#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch.utils.data as data
import numpy as np
import torch


class KfoldTorchDataset(data.Dataset):
    
    def __init__(self, inputs, labels):
        self.samples = np.array(inputs)
        self.labels = torch.Tensor(labels).long()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = self.samples[idx]
        x =  np.expand_dims(x, axis=0)
        y = self.labels[idx]
        x = torch.Tensor(x)
        return x, y
        
        
        