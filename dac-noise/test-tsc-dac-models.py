#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 19:42:56 2020

@author: daniyalusmani1

1. Load the training data with noise that was used for training. (The samples should not be shuffled)
2. Load the best model from the training phase
3. Check which samples are allotted abstained class
4. Cross reference the abstained samples with samples which were flipped when noise was generated.

Run this process for each dataset and for each noise level 3 to 5 times and get the average
"""


from __future__ import print_function

import argparse

parser = argparse.ArgumentParser(description='PyTorch testing for deep abstaining classifiers',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--datadir',  type=str, required=True, help='data directory')
parser.add_argument('--results_dir',  type=str, required=False, help='results directory')
parser.add_argument('--exp_name',  type=str, required=False, help='experiment name (model would be fetched from this)')
parser.add_argument('--iteration', default=0, type=int, help='iteration for noise dataset')
parser.add_argument('--dataset', default='UCRArchive_2018', type=str, help='dataset = [ucr-archive / ai_crop]')
parser.add_argument('--net_type', default='inception', type=str, help='[tsc-lstm / inception-simple]')
parser.add_argument('--loss_fn', default=True, type=bool, help='adds class for abstention')
parser.add_argument('--checkpoint_path', default='', type=str, help='path to add model checkpoint from')
parser.add_argument('--noise_percentage', default=0, type=float, help='noise percentage')
parser.add_argument('--batch_size', default=128, type=int, help='batch size of training')
parser.add_argument('--test_batch_size', default=128, type=int, help='batch size for testing')
parser.add_argument('--depth', default=2, type=int, help='depth of lstm')


args = parser.parse_args()

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable

import os
import sys
import time
import datetime

import numpy as np

from utils import gpu_utils, datasets, label_noise, kfold_torch_dataset

from networks import lstm, inceptionNet
from networks import config as cf


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('\n[Phase 1] : Data Preparation')
# get data for the provided noise percentage
trainset, testset, num_classes, series_length, train_sampler, valid_sampler = datasets.get_data(args)

# get data for simple dataset
#args.noise_percentage = 0
#args.iteration = 0
trainset_no_noise, testset_no_noise, num_classes_no_noise, series_length_no_noise, train_sampler_no_noise, valid_sampler_no_noise = datasets.get_data(args)


# TODO: have to generate val and training for ai_crop dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

trainloader_no_noise = torch.utils.data.DataLoader(trainset_no_noise, batch_size=args.batch_size, shuffle=False, num_workers=2)

sys.stdout.flush()

def getNetwork(args):
	if args.loss_fn is None:
		extra_class = 0
	else:
		extra_class = 1
    
	if (args.net_type == 'tsc-lstm'):
        # the input dimension has dimension of 1
		net = lstm.TSCLSTM(1,series_length, args.depth, num_classes+extra_class)
		# file_name = 'tsc-lstm-'+str(args.depth)+'x'+str(args.widen_factor)
		file_name = 'tsc-lstm-'
	elif (args.net_type == 'inception_simple'):
        # the input dimension has dimension of 1
		net = inceptionNet.InceptionNet(num_classes+extra_class)
		# file_name = 'inception-simple-'+str(args.depth)+'x'+str(args.widen_factor)
		file_name = 'inception-simple-'
	return net.to(device), file_name
    

def loadModel():
    if torch.cuda.is_available():
        checkpoint = torch.load(args.checkpoint_path)
    else:
        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))

    model = checkpoint['net']
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    return model
    

def evalDataset(loader):
    total_predicted = np.array([])
    total_targets = np.array([])
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = Variable(inputs.to(device)), Variable(targets.to(device))
        if (args.net_type == 'tsc-lstm'): # update input to match lstm
            inputs = inputs.view(-1, series_length, 1) # input dimensions for time series data set are 1
		# print(inputs.shape)
        outputs = model(inputs)               # Forward Propagation
        # probs = outputs.cpu().detach().numpy()  
        _, predicted = torch.max(outputs.data, 1)
        total_predicted = np.append(total_predicted, predicted.cpu().detach().numpy())
        total_targets = np.append(total_targets, targets.cpu().detach().numpy())
    return total_predicted, total_targets

def compareLabels(noisyLabels, origLabels):
    noise = torch.from_numpy(noisyLabels).to(device)
    orig = torch.from_numpy(origLabels).to(device)
    
    label_diff = len(origLabels) - noise.eq(orig).cpu().sum().data.item()
    
    diff_index = np.invert(noise.eq(orig).cpu().detach().numpy()).nonzero()[0]
    
    noise_diff_labels = noise[noise.eq(orig) == False].cpu().detach().numpy()
    orig_diff_labels = orig[orig.eq(noise) == False].cpu().detach().numpy()
    
    assert len(diff_index) == len(noise_diff_labels), "The length is not the same"
    assert len(diff_index) == len(orig_diff_labels), "The length is not the same"

    return label_diff, diff_index, noise_diff_labels, orig_diff_labels

# The combined_labels is a 2D array of the following order
#  [orig_labels, train_targets, train_predicted] 
def compareAbstained(combined_labels):
    abstained_class = float(num_classes)
    trained = combined_labels[:][2]
    trained_abstained = trained == abstained_class
    print(trained_abstained)
    print(combined_labels.shape)
    all_abstained = combined_labels[:,trained_abstained]
    print("total abstained: %i"%(len(all_abstained[0,:])))
    print(all_abstained)
    # print(combined_labels[:][2])
    correctly_abstained = all_abstained[:,all_abstained[0,:] != all_abstained[1,:]]
    wrongly_abstained = all_abstained[:,all_abstained[0,:] == all_abstained[1,:]]
    print("correctly abstained: %i"%(len(correctly_abstained[0,:])))
    print(correctly_abstained)  
    print("wrongly abstained: %i"%(len(wrongly_abstained[0,:])))
    print(wrongly_abstained)
    return all_abstained, correctly_abstained, wrongly_abstained    

net , file_name = getNetwork(args) 
model = loadModel()
train_predicted, train_targets = evalDataset(trainloader)
#test_predicted = evalDataset(testloader)
_, orig_labels = evalDataset(trainloader_no_noise)

combined_labels_arr = np.array([orig_labels, train_targets, train_predicted])
print(combined_labels_arr)

compareLabels(train_predicted, train_targets)
label_diff, diff_index, noise_diff_labels, orig_diff_labels = compareLabels(train_targets, orig_labels)
all_abstained, correctly_abstained, wrongly_abstained = compareAbstained(combined_labels_arr)









