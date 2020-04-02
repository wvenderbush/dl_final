# SOURCE: https://algorithmia.com/blog/convolutional-neural-nets-in-pytorch 

# 1) decide on size of the embedding layer
# 2) figure out how to download images for training

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn.functional as F
from data_loader import PaintingDataset


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# A) LOADING THE DATA

#The compose function allows for multiple transforms
#transforms.ToTensor() converts our PILImage to a tensor of shape (C x H x W) in the range [0,1]
#transforms.Normalize(mean,std) normalizes a tensor to a (mean, std) for (R, G, B)
print(1)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
print(2)
dataset = PaintingDataset(transform=transform)
print(3)
train_set = dataset #TODO #torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)
print(5)
test_set = dataset #TODO #torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)
print(6)
classes = ('1401-1450', '1451-1500', '1501-1550', '1551-1600',
           '1601-1650', '1651-1700', '1701-1750', '1751-1800', '1801-1850', '1851-1900')

from torch.utils.data.sampler import SubsetRandomSampler

n_samples = len(dataset)

#Training
n_training_samples = n_samples*0.1 # TODO
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
print(7)
#Validation
n_val_samples = n_samples*0.01 # TODO
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))
print(8)
#Test
n_test_samples = n_samples*0.01 # TODO
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))
print(9)
# B) CLASS FOR CNN!!!


class SimpleCNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self):
        print('a')
        super(SimpleCNN, self).__init__()
        print('b')
        
        #512x512x3
        #Input channels = 3, output channels = 64
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        print('c')
        #512x512x64
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        print('d')
        
        #256x256x64
        #self.conv2 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        #512x512x64
        #self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        
        
        ###256x256x64 input features, 64x256 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(64 * 256 * 256, 64) #* 256)
        print('e')
        
        ###256x64 input features, 64 output features (see sizing flow below)
        # self.fc2 = torch.nn.Linear(64 * 256, 64)
        # print('f')
        #64 input features, 10 output features for our 10 defined classes
        self.fc3 = torch.nn.Linear(64, 10)
        print('g')
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 512, 512) to (64, 512, 512)
        x = F.relu(self.conv1(x))
        
        #Size changes from (64, 512, 512) to (64, 256, 256)
        x = self.pool(x)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (64, 256, 256) to (1, 64x256x256)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 64 * 256 * 256)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 64x256x256) to (1, 64x256)
        x = F.relu(self.fc1(x))
        
        #Computes the activation of the second fully connected layer
        #Size changes from (1, 64x256) to (1, 64)
        # x = F.relu(self.fc2(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc3(x)
        return(x)

print(10)
def outputSize(in_size, kernel_size, stride, padding):

    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1

    return(output)
  
# C) TRAINING THE NEURAL NET

#DataLoader takes in a dataset and a sampler for loading (num_workers deals with system level memory) 
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=2)
    return(train_loader)

#Test and validation loaders have constant batch sizes, so we can define them directly
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)
val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)

import torch.optim as optim

def createLossAndOptimizer(net, learning_rate=0.001):
    
    #Loss function
    loss = torch.nn.CrossEntropyLoss()
    
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    return(loss, optimizer)

import time
print(11)
def trainNet(net, batch_size, n_epochs, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    
    #Time for printing
    training_start_time = time.time()
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader, 0):
            
            #Get inputs
            inputs, labels = data
            
            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            running_loss += loss_size.data[0]
            total_train_loss += loss_size.data[0]
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in val_loader:
            
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data[0]
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
print(12)
CNN = SimpleCNN()
print(13)
trainNet(CNN, batch_size=32, n_epochs=5, learning_rate=0.001)
print(14)
  