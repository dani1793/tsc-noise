#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 23:42:17 2020

@author: daniyalusmani1
"""

#import InceptionNetResidualBlock
from networks import InceptionNetResidualBlock

# import inceptionModule
from networks import inceptionModule
from torch import nn

class InceptionBlockWithResidual(InceptionNetResidualBlock.InceptionNetResidualBlock):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion =1, downsampling = 1, *args, **kwargs)
        self.blocks = nn.Sequential(inceptionModule.InceptionModule(in_channels,64,1,8,32,32,True), 
                                    inceptionModule.InceptionModule(32,64,1,18,32,32,True)
                                    )
        #self.blocks = nn.Sequential(inceptionModule.InceptionModule(in_channels,64,1,41,32,32,True), 
         #                           inceptionModule.InceptionModule(32,64,1,41,32,32,True)
          #                          )
    
    def forward(self, x):
        x = self.blocks(x);
        return x;

if __name__ == '__main__':
    net=InceptionBlockWithResidual(32,64) 
    print(net)
    #y = net(Variable(torch.randn(64, 1, 64)))
    #print(y.size())           