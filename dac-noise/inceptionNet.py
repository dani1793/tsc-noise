# -*- coding: utf-8 -*-
from torch.nn import Module, Sequential, Linear, Softmax

from networks import inceptionModule;
from networks import InceptionBlockWithResidual;
import torch
from torch.autograd import Variable    

class InceptionNet(Module):
    def __init__(self, nb_classes, verbose=False, build=True, batch_size=64,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500, stride=1, activation='linear'):
       # self.output_directory = output_directory
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.depth = depth;
        self.nb_classes = nb_classes;
        
        self.stride = stride;
        
        super().__init__()
  

        net = [InceptionBlockWithResidual.InceptionBlockWithResidual(1,32, 8, 18),
                              inceptionModule.InceptionModule(32,64,1,24,32,32,True),
                              InceptionBlockWithResidual.InceptionBlockWithResidual(32,32, 20, 12),
                              inceptionModule.InceptionModule(32,32,1,8,32,32,True)]
        
        #net = [InceptionBlockWithResidual.InceptionBlockWithResidual(1,32),
         #                     inceptionModule.InceptionModule(32,64,1,41,32,32,True),
          #                    InceptionBlockWithResidual.InceptionBlockWithResidual(32,32),
           #                   inceptionModule.InceptionModule(32,32,1,41,32,32,True)]
    
        self.net = Sequential(*net);
        
        self.output_layer = Sequential(Linear(32, nb_classes), Softmax(1))
        
        
    def forward(self, x):
        # for d in range(self.depth):
        x = self.net(x)
        # global average pooling for 1D
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = self.output_layer(x)
        return x

if __name__ == '__main__':
    net=InceptionNet(8) 
    print(net)
    y = net(Variable(torch.randn(64, 1, 64)))
    print(y.size())  
    print("Learnable parameter count : ", sum(p.numel() for p in net.parameters() if p.requires_grad))
