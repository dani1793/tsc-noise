# -*- coding: utf-8 -*-
from torch.nn import Module, Identity, Conv1d, MaxPool1d, BatchNorm1d, ReLU
import torch

class InceptionModule(Module):
    
        def canUseBottleneck(self, inputTensor):
            return self.use_bottleneck and int(inputTensor.shape[-1]) > 1;
        
        def __init__(self, in_channels, out_channels, stride, kernel_size, nb_filters, bottleneck_size, use_bottleneck):
            super().__init__()
            
            self.bottleneck_size = bottleneck_size;
            self.stride = stride;
            self.kernel_size = kernel_size;
            self.nb_filters = nb_filters;
            self.use_bottleneck = use_bottleneck;
            
            self.input = Identity()
            
            self.bottleneckInput = Conv1d(in_channels, self.bottleneck_size, kernel_size=1, padding=0, bias=False);
            
            # kernel_size_s = [3, 5, 8, 11, 17]
            kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
            #print("kernel size");
            #print(kernel_size_s)

            self.conv_list = []
            self.temp_input = self.bottleneck_size
            for i in range(len(kernel_size_s)):
                setattr(self, "cov_parallel_%d" % kernel_size_s[i],Conv1d(self.temp_input, self.nb_filters, kernel_size=kernel_size_s[i], stride=self.stride, padding=0,bias=False))
                self.temp_input= self.nb_filters;
                self.conv_list.append(getattr(self,"cov_parallel_%d" % kernel_size_s[i]))   
            self.max_pool = MaxPool1d(kernel_size=3, stride=self.stride, padding=0)
            
            self.conv = Conv1d(self.bottleneck_size, self.nb_filters, kernel_size=1, padding=0, bias=False)
            
            self.bn = BatchNorm1d(self.nb_filters)
        
        def forward(self, x):
            if self.canUseBottleneck(x):
                x = self.bottleneckInput(x);
            parallelLayers = [];
            
            for layer in self.conv_list:
                parallelLayers.append(layer(x));
            res = self.max_pool(x);
            res = self.conv(res);
            
            parallelLayers.append(res)
            
            x = torch.cat(parallelLayers, 2)
                           
            x = self.bn(x);
            
            x = ReLU()(x);
            ##print("final output");
            #print(x.shape)      
            return x;
            
            
            