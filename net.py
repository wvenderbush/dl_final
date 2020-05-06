# 1) decide on size of the embedding layer
# 2) figure out how to download images for training

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


from torch.autograd import Variable
import torch.nn.functional as F
from data_loader import PaintingDataset
from torch.utils.data import random_split


# seed = 42
# np.random.seed(seed)
# torch.manual_seed(seed)

# A) LOADING THE DATA

#The compose function allows for multiple transforms
#transforms.ToTensor() converts our PILImage to a tensor of shape (C x H x W) in the range [0,1]
#transforms.Normalize(mean,std) normalizes a tensor to a (mean, std) for (R, G, B)
# print(1)
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# print(2)
dataset = PaintingDataset(transform=transform)

train_set, val_set, test_set = random_split(dataset, (int(len(dataset)*0.01),
                                                      int(len(dataset)*0.01),
                                                      len(dataset)-int(len(dataset)*0.01)-int(len(dataset)*0.01)))

# print(type(train_set))
# print(3)
#train_set = dataset #TODO #torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)
# print(5)
#test_set = dataset #TODO #torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)
# print(6)

# trnsfrm = transforms.Resize((512, 512))
# import matplotlib.pyplot as plt
# # Check for scaling
# plt.figure()

# plt.subplot(1, 2, 1)

# plt.imshow(dataset[0][0]) # PYPLOT WANTS AN img_obj as in __getitem__

# plt.subplot(1, 2, 2)
# sampleTransformed = trnsfrm(dataset[0][0])
# plt.imshow(sampleTransformed)


# plt.show()

classes = {'1401-1450', '1451-1500', '1501-1550', '1551-1600',
           '1601-1650', '1651-1700', '1701-1750', '1751-1800', '1801-1850', '1851-1900'}

from torch.utils.data.sampler import SubsetRandomSampler

n_samples = len(dataset)

#Training
#n_training_samples = n_samples*0.1 # TODO
train_sampler = SubsetRandomSampler(np.arange(len(train_set), dtype=np.int64))
# print(7)
#Validation
#n_val_samples = n_samples*0.01 # TODO
val_sampler = SubsetRandomSampler(np.arange(len(val_set), dtype=np.int64))
# print(8)
#Test
#n_test_samples = n_samples*0.01 # TODO
test_sampler = SubsetRandomSampler(np.arange(len(test_set), dtype=np.int64))
# print(9)
# B) CLASS FOR CNN!!!


class SimpleCNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self):
        # print('a')
        super(SimpleCNN, self).__init__()
        # print('b')
        
        #256x256x3
        #Input channels = 3, output channels = 64
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # print('c')
        #256x256x64
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # print('d')
        
        #256x256x64
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        #512x512x64
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        
        
        ###128x128x64 input features, 64x128 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(128 * 64 * 64, 128)
        # print('e')
        
        ###256x64 input features, 64 output features (see sizing flow below)
        self.fc2 = torch.nn.Linear(128, 64)
        # print('f')
        ###FBAJDBCKJASVCDIJKNDLKCBADBCKASBJWB C   TODO   CHANGE THIS BACK
        #64 input features, 18* output features for our 18* defined classes
        self.fc3 = torch.nn.Linear(64, 10)
        # print('g')
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 512, 512) to (64, 512, 512)
        x = F.relu(self.conv1(x))
        
        #Size changes from (64, 512, 512) to (64, 256, 256)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))

        x = self.pool2(x)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (64, 256, 256) to (1, 64x256x256)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 128 * 64 * 64)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 64x256x256) to (1, 64x256)
        x = F.relu(self.fc1(x))
        
        #Computes the activation of the second fully connected layer
        #Size changes from (1, 64x256) to (1, 64)
        x = F.relu(self.fc2(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc3(x)
        return(x)

# print(10)
def outputSize(in_size, kernel_size, stride, padding):

    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1

    return output
  
# C) TRAINING THE NEURAL NET

#DataLoader takes in a dataset and a sampler for loading (num_workers deals with system level memory) 
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=2)
    return train_loader

#Test and validation loaders have constant batch sizes, so we can define them directly
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2) #
val_loader = torch.utils.data.DataLoader(train_set, batch_size=32, sampler=val_sampler, num_workers=2) # TODO BATCH SIZE was 128

import torch.optim as optim

def createLossAndOptimizer(net, learning_rate=0.001, weight_decay=0.0):
    
    #Loss function
    loss = torch.nn.CrossEntropyLoss()
    
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    return(loss, optimizer)

# Dict = dict({'1401-1450': torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).double(),
#             '1451-1500': torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).double(),
#             '1501-1550': torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).double(),
#             '1551-1600': torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).double(),
#             '1601-1650': torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).double(),
#             '1651-1700': torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).double(),
#             '1701-1750': torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).double(),
#             '1751-1800': torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).double(),
#             '1801-1850': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).double(),
#             '1851-1900': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).double()}) 


# Dict = dict({'1401-1450': torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).double(),
#             '1451-1500': torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).double(),
#             '1501-1550': torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).double(),
#             '1551-1600': torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).double(),
#             '1601-1650': torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).double(),
#             '1651-1700': torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).double(),
#             '1701-1750': torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).double(),
#             '1751-1800': torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).double(),
#             '1801-1850': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).double(),
#             '1851-1900': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).double()}) 


## TODO
Dict = {'1401-1450': 0,
             '1451-1500': 1,
             '1501-1550': 2,
             '1551-1600': 3,
             '1601-1650': 4,
             '1651-1700': 5,
             '1701-1750': 6,
             '1751-1800': 7,
             '1801-1850': 8,
             '1851-1900': 9}

import time
# print(11)
def trainNet(net, batch_size, n_epochs, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    train_loader = get_train_loader(batch_size)
    # print(15)
    n_batches = len(train_loader)

    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    # print(16)
    #Time for printing
    training_start_time = time.time()
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        # print (17)
        running_loss = 0.0
        print_every = 3#n_batches // 20
        start_time = time.time()
        total_train_loss = 0
        # print(19)
        # print(len(train_loader))
        total_tested = 0
        total_correct = 0
        for i, data in enumerate(train_loader):
            # print(i)
            #Get inputs
            inputs, labels = data
            one_hot_labels = []
            for label in labels:
                try:
                    one_hot_labels.append(Dict[label])
                except KeyError:
                    continue

            # for i in range(len(one_hot_labels)):
                # one_hot_labels[i] = one_hot_labels[i].unsqueeze(0)
            # print(one_hot_labels)

            # labels = torch.cat(one_hot_labels, 0)
            labels = torch.tensor(one_hot_labels)

            # print(19.1)
            # print(inputs, flush=True)
            # print(labels, flush=True)
            
            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            # print(labels)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            # print(19.2)

            outputs = net(inputs)
            # print(torch.tensor([int(x.max(0)[1]) for x in outputs]))
            # print(labels)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()


            predicted_labels = torch.tensor([int(x.max(0)[1]) for x in outputs])
            true_labels = labels
            for ii in range(len(predicted_labels)):
                total_tested += 1
                if predicted_labels[ii] == true_labels[ii]:
                    total_correct += 1
            
            
            #Print statistics
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()
            # print(19.3)
            
            #Print every 10th batch of an epoch TODO
            if (i + 1) % (print_every + 1) == 0:
               print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                       epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
               #Reset running loss and time
               running_loss = 0.0
               start_time = time.time()

        print("Accuracy of test set: ", total_correct/total_tested)

        # print(20, flush=True)
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        # j=0
        total_tested = 0
        total_correct = 0
        for inputs, labels in val_loader:
            # print(len(val_loader))
            # print(j)
            # j += 1

            one_hot_labels = []
            for label in labels:
                try:
                    one_hot_labels.append(Dict[label])
                except KeyError:
                    continue
            # print(20.1)
            labels = torch.tensor(one_hot_labels)
            # print(20.2)
            
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            # print(20.3)
            
            #Forward pass
            val_outputs = net(inputs)
            # print(20.4)
            predicted_labels = torch.tensor([int(x.max(0)[1]) for x in val_outputs])

            true_labels = labels
            for ii in range(len(predicted_labels)):
                total_tested += 1
                if predicted_labels[ii] == true_labels[ii]:
                    total_correct += 1

            val_loss_size = loss(val_outputs, labels)
            # print(20.5)
            total_val_loss += val_loss_size.item()
            # print(20.6)
            # print(predicted_labels, true_labels)
        print("Accuracy of validation set: ", total_correct/total_tested)
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
# print(12)
CNN = SimpleCNN()
# print(13)
# TODO
trainNet(CNN, batch_size=32, n_epochs=5, learning_rate=0.001) # TODO: Batchsize original was 32
# print(14)

torch.save(CNN.state_dict(), './CNN.pth')
  
