#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 23:06:59 2020

@author: daniyalusmani1
"""


from utils import ucr_archive, crop_tsc_balanced_2015
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler


if __name__ == '__main__':
    final = pd.DataFrame()
    
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = "precision"
    res['accuracy'] = "accuracy"
    
    res['accuracy_val'] = "accuracy_val"

    res['recall'] = "recall"
    res['duration'] = "duration"
    res["k-fold"] = 1
    final = pd.concat([final, res])
    
    
    res2 = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res2['precision'] = "precision"
    res2['accuracy'] = "accuracy"
    
    res2['accuracy_val'] = "accuracy_val"

    res2['recall'] = "recall"
    res2['duration'] = "duration"
    res2["k-fold"] = 2
    final = pd.concat([final, res2])
    

    print(pd.concat([res,res2]))
    
    
'''
 #   dataset = ucr_archive.UCRArchive('data/UCRArchive_2018', 'SmoothSubspace')
    dataset = crop_tsc_balanced_2015.CropTscBalanced2015('data/ai_crop', 'crop_tsc_balanced_filled_2015.csv')
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        
    dataloader = DataLoader(dataset, batch_size=120, shuffle=True, num_workers=2)
    for i, batch in enumerate(dataloader):
        print(i, batch)
        print(batch[0].shape)
'''