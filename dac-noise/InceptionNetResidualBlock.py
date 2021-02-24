#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 01:15:34 2020

@author: daniyalusmani1
"""

from torch import nn;


from networks import ResidualBlock
from networks.utils import activation_func



class InceptionNetResidualBlock(ResidualBlock.ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling = expansion, downsampling
        self.activate = activation_func('relu')
        self.shortcut = nn.Sequential(
            nn.Conv1d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm1d(self.expanded_channels)) if self.should_apply_shortcut else None 
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels
    

    # TODO: add forward method

        

if __name__ == '__main__':
    net=InceptionNetResidualBlock(32,64) 
    print(net)
    #y = net(Variable(torch.randn(64, 1, 64)))
    #print(y.size())  