"""
load and prepare various datasets
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
try:
	import stl10_c
except ModuleNotFoundError:
	from . import stl10_c

try:
	import ucr_archive
except ModuleNotFoundError:
	from . import ucr_archive
    

try:
	import crop_tsc_balanced_2015
except ModuleNotFoundError:
	from . import crop_tsc_balanced_2015
    
import sys


def get_data_samples( trainIndices, valIndices):
    train_sampler = SubsetRandomSampler(trainIndices)
    valid_sampler = SubsetRandomSampler(valIndices)
    return train_sampler, valid_sampler

def getdatasetDict(args):
    
    if args.dataset == 'ucr-archive':
        return ucr_archive.getSimpleUCRArchive(args.datadir, 'Crop', noise = args.noise_percentage, transform=None)

def get_data(args,):
    
	train_sampler = None
	valid_sampler = None

	if args.dataset == 'cifar10' or args.dataset == 'cifar100':
		cifar_mean = {
		    'cifar10': (0.4914, 0.4822, 0.4465),
		    'cifar100': (0.5071, 0.4867, 0.4408),
		}

		cifar_std = {
		    'cifar10': (0.2023, 0.1994, 0.2010),
		    'cifar100': (0.2675, 0.2565, 0.2761),
		}

		cifar_transform_train = transforms.Compose([
		    transforms.RandomCrop(32, padding=4),
		    transforms.RandomHorizontalFlip(),
		    transforms.ToTensor(),
		    transforms.Normalize(cifar_mean[args.dataset], cifar_std[args.dataset]),
		]) # meanstd transformation

		cifar_transform_test = transforms.Compose([
		    transforms.ToTensor(),
		    transforms.Normalize(cifar_mean[args.dataset], cifar_std[args.dataset]),
		])



	### STL-10/STL-labeled/STL-C transforms
	transform_train_stl10 = transforms.Compose([
			transforms.RandomCrop(96,padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			#calculated using snippet above
			transforms.Normalize((0.447, 0.44, 0.4), (0.26, 0.256, 0.271))
		])

	transform_test_stl10=transforms.Compose([
		transforms.ToTensor(),
		#calculated using snippet above
		transforms.Normalize((0.447, 0.44, 0.405), (0.26, 0.257, 0.27))
	])

	### Tiny Imagenet 200 # copied from STL-10 for now.
	transform_train_tin200 = transforms.Compose([
			transforms.RandomCrop(96,padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			#calculated using snippet above
			transforms.Normalize((0.447, 0.44, 0.4), (0.26, 0.256, 0.271))
		])

	transform_test_tin200=transforms.Compose([
		transforms.ToTensor(),
		#calculated using snippet above
		transforms.Normalize((0.447, 0.44, 0.405), (0.26, 0.257, 0.27))
	])


	# #original, mostly untransformed train set.
	# transform_train_orig_stl10 = transforms.Compose([
	# 	transforms.ToTensor(),
	# 	#calculated using snippet above
	# 	transforms.Normalize((0.447, 0.44, 0.405), (0.26, 0.257, 0.27))
	# ])

	#### MNIST Transforms
	#from https://github.com/pytorch/examples/blob/master/mnist/main.py
	mnist_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))])

	fashion_mnist_train_transform = transforms.Compose([
		transforms.RandomCrop(28,padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.2868,),(0.3524,))

		 ])

	fashion_mnist_test_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.2860,),(0.3530,))

		 ])

	if(args.dataset == 'cifar10'):
		print("| Preparing CIFAR-10 dataset...")
		sys.stdout.write("| ")
		trainset = torchvision.datasets.CIFAR10(root=args.datadir, train=True, 
			download=False, transform=cifar_transform_train)
		testset = torchvision.datasets.CIFAR10(root=args.datadir, train=False,
			download=False, transform=cifar_transform_test)
		num_classes = 10

	elif(args.dataset == 'cifar100'):
	    print("| Preparing CIFAR-100 dataset...")
	    sys.stdout.write("| ")
	    trainset = torchvision.datasets.CIFAR100(root=args.datadir, train=True,
	    	download=False, transform=cifar_transform_train)
	    testset = torchvision.datasets.CIFAR100(root=args.datadir, train=False, 
	    	download=False, transform=cifar_transform_test)
	    num_classes = 100

	elif(args.dataset == 'mnist'):
		print("| Preparing MNIST dataset...")
		sys.stdout.write("| ")
		trainset = torchvision.datasets.MNIST(root=args.datadir, train=True, 
			download=True, transform=mnist_transform)
		testset = torchvision.datasets.MNIST(root=args.datadir, train=False, 
			download=True, transform=mnist_transform)
		num_classes = 10

	elif(args.dataset == 'fashion'):
		print("| Preparing Fashion MNIST dataset...")
		sys.stdout.write("| ")
		trainset = torchvision.datasets.FashionMNIST(root=args.datadir, train=True, 
			download=False, transform=fashion_mnist_train_transform)
		testset = torchvision.datasets.FashionMNIST(root=args.datadir, train=False, 
			download=False, transform=fashion_mnist_test_transform)
		num_classes = 10

		#below for a one-time calculation of mean and std.
		# print(list(trainset.train_data.size()))
		# print(trainset.train_data.float().mean()/255)
		# print(trainset.train_data.float().std()/255)	

		# print(list(testset.test_data.size()))
		# print(testset.test_data.float().mean()/255)
		# print(testset.test_data.float().std()/255)	

		# sys.exit(0)


	elif (args.dataset == 'stl10-labeled'):
		print("| Preparing STL10-labeled dataset...")
		
		trainset = torchvision.datasets.STL10(root=args.datadir, 
			split='train', download=False, transform=transform_train_stl10)
		testset = torchvision.datasets.STL10(root=args.datadir,
			split='test', download=False, transform=transform_test_stl10)
		num_classes = 10

	elif (args.dataset == 'stl10-c'):
		print("| Preparing STL10-C dataset...")
		trainset = stl10_c.STL10_C(root=args.datadir, 
			#split='train', transform=transform_train_stl10, train_list=[[args.train_x,''],[args.train_y,'']])
			split='train', transform=transform_train_stl10, train_list=[[args.train_x,''],[args.train_y,'']])
		testset = stl10_c.STL10_C(root=args.datadir,
			split='test', transform=transform_test_stl10, test_list = [[args.test_x,''],[args.test_y,'']])
		num_classes = 10

	elif (args.dataset == 'tin200'):
		print("| Preparing TinyImagenet-200 dataset...")
		import tiny_imagenet_200 as tin200
		trainset = tin200.TINY_IMAGENET_200(root=args.datadir, 
			#split='train', transform=transform_train_stl10, train_list=[[args.train_x,''],[args.train_y,'']])
			split='train', transform=transform_train_tin200, train_list=[[args.train_x,''],[args.train_y,'']])
		testset = tin200.TINY_IMAGENET_200(root=args.datadir,
			split='test', transform=transform_test_tin200, test_list = [[args.test_x,''],[args.test_y,'']])
		num_classes = 11
	elif (args.dataset == 'ucr-archive'):
		print("| Preparing UCRArchive dataset...")
		#trainset = ucr_archive.UCRArchive(args.datadir, 'SmoothSubspace', datasetType = 'TRAIN', transform=None)
		#testset = ucr_archive.UCRArchive(args.datadir, 'SmoothSubspace', datasetType = 'TEST', transform=None)        
		#num_classes = 3
		#series_length = 15
        
        
		#trainset = ucr_archive.UCRArchive(args.datadir, 'Chinatown', datasetType = 'TRAIN', noise = args.noise_percentage, transform=None)
		#testset = ucr_archive.UCRArchive(args.datadir, 'Chinatown', datasetType = 'VAL', noise = args.noise_percentage, transform=None)              
		#num_classes = 10       
		#series_length = 24

        
		trainset = ucr_archive.UCRArchiveNoisyVal(args.datadir, 'Crop', iteration= args.iteration, datasetType = 'TRAIN', noise = args.noise_percentage, transform=None)
		# testset = ucr_archive.UCRArchive(args.datadir, 'Crop', datasetType = 'VAL', noise = args.noise_percentage, transform=None) Validation set is always non noisy      
		# testset = ucr_archive.UCRArchive(args.datadir, 'Crop', iteration= args.iteration, datasetType = 'VAL', noise = , transform=None)
		testset = ucr_archive.UCRArchiveNoisyVal(args.datadir, 'Crop', iteration= args.iteration, datasetType = 'VAL', noise = args.noise_percentage, transform=None)

		num_classes = 24 
		series_length = 46
        
	elif args.dataset == 'ai_crop':
        
		trainset = crop_tsc_balanced_2015.CropTscBalanced2015(args.datadir, args.dataset,  9, 8, iteration= args.iteration, datasetType = 'TRAIN',transform=None)
		testset = crop_tsc_balanced_2015.CropTscBalanced2015(args.datadir, args.dataset,  9, 8, iteration= args.iteration, datasetType = 'VAL',transform=None)
        
		num_classes = 2 
		series_length = 8
        
	elif (args.dataset == 'crop_tsc_balanced_filled_2015.csv'):
		print(args.dataset)
		trainset = crop_tsc_balanced_2015.CropTscBalanced2015(args.datadir, args.dataset, 13, 12, transform=None)
		testset = trainset       


        # Creating data indices for training and validation splits:
		dataset_size = len(trainset)
		indices = list(range(dataset_size))
		shuffle_dataset = True
		validation_split = .2
		random_seed= 42
		split = int(np.floor(validation_split * dataset_size))
		if shuffle_dataset :
  		    np.random.seed(random_seed)
  		    np.random.shuffle(indices)
		train_indices, val_indices = indices[split:], indices[:split]
          
      # Creating PT data samplers and loaders:
		train_sampler = SubsetRandomSampler(train_indices)
		valid_sampler = SubsetRandomSampler(val_indices)            
            
		num_classes = 2 
		series_length = 12
	elif (args.dataset == 'crop_tsc_balanced_imputed_2015.csv' or args.dataset == 'crop_tsc_balanced_imputed_PCA_2015.csv'):
		print(args.dataset)
		trainset = crop_tsc_balanced_2015.CropTscBalanced2015(args.datadir, args.dataset, 9, 8, transform=None)
		testset = trainset       
		validation_split = .5
		shuffle_dataset = True
		random_seed= 42

        # Creating data indices for training and validation splits:
		dataset_size = len(trainset)
		indices = list(range(dataset_size))
		split = int(np.floor(validation_split * dataset_size))
		if shuffle_dataset :
		    np.random.seed(random_seed)
		    np.random.shuffle(indices)
		train_indices, val_indices = indices[split:], indices[:split]
        
        # Creating PT data samplers and loaders:
		train_sampler = SubsetRandomSampler(train_indices)
		valid_sampler = SubsetRandomSampler(val_indices)
        
		num_classes = 2 
		series_length = 8

            
	else:
		print("Unknown data set")
		sys.exit(0)

	return trainset, testset, num_classes, series_length, train_sampler, valid_sampler
