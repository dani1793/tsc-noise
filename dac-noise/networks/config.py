############### Pytorch CIFAR configuration file ###############
import math

start_epoch = 1
num_epochs = 200
batch_size = 128
optim_type = 'SGD'

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def learning_rate(init, epoch):
    
   #return init;
   optim_factor = 0

   #print("changing learning rate for inception Time CROP");
   # print("keeping it constant thoughout the training process");
  # print("using config steps for inceptiom time")
   if (epoch > 300):
       optim_factor = 5;
   elif (epoch > 280):
       optim_factor = 5;
   elif (epoch > 250):
       optim_factor = 5;
   elif (epoch > 220):
       optim_factor = 5;
   elif (epoch > 200):
       optim_factor = 5;
   elif (epoch > 150):
       optim_factor = 5;
   elif (epoch > 80):
       optim_factor = 3;
   elif (epoch > 50):
       optim_factor = 1
   elif (epoch > 30):
       optim_factor = 1;
   elif (epoch > 20):
       optim_factor = 1;
   elif (epoch > 10):
       optim_factor = 1;
       

    
   
   '''   #print("changing learning rate for lstm");
# FOR LSTM UCR CROP

   if (epoch > 3000):
       optim_factor = 7 #4 #7
   elif (epoch > 1900):
       optim_factor = 6 #4 #7
   elif (epoch > 1700):
       optim_factor = 5 #3 #6
   elif (epoch > 1500):
       optim_factor = 5 #2 #5
   elif (epoch > 1200):
       optim_factor = 5 #1 #4
   elif (epoch > 950):
       optim_factor = 5 #3        
   elif (epoch > 750):
       optim_factor = 5 #2
   elif (epoch > 550):
       optim_factor = 5 #1 #5
   elif (epoch > 300):
       optim_factor = 5 #1#4
   elif(epoch > 240):
       optim_factor = 4 #3
   elif(epoch > 120):
       optim_factor = 4 #3
   elif(epoch > 80):
       optim_factor = 2 #2
   elif(epoch > 45):
       optim_factor = 1 #2
    #elif(epoch > 160):
    #    optim_factor = 1'''
  
   #return init*math.pow(0.2, optim_factor)
   return init*math.pow(0.5, optim_factor) # for inception Time crop
   #return init*math.pow(0.2, optim_factor) # for inception Time crop      

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
