#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 01:15:13 2020

@author: daniyalusmani1
"""

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

from torch.nn import Module, Identity, Conv1d, MaxPool1d, BatchNorm1d, ReLU
import torch
from torch.autograd import Variable


xtorch = [Variable(torch.randn(64, 1, 57)), Variable(torch.randn(64, 1, 61)), Variable(torch.randn(64, 1, 63))]
concatTorchTensor = torch.cat(xtorch, 2)                                                                                      
print(concatTorchTensor.shape)


xkeras = [np.arange(64 * 1 * 57).reshape(64, 1, 57), np.arange(64 * 1 * 61).reshape(64, 1, 61), np.arange(64 * 1 * 63).reshape(64, 1, 63)] 
concatKerasTensor = tf.keras.layers.Concatenate(axis=2)(xkeras)
print(concatKerasTensor.shape)
assert concatTorchTensor.shape == concatKerasTensor.shape, "The shape is not equal for concatenation process"