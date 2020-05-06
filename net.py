import numpy as np
import torch
import torchvision
import os
import torchvision.transforms as transforms


from torch.autograd import Variable
import torch.nn.functional as F
from data_loader import PaintingDataset
from torch.utils.data import random_split

# A) LOADING THE DATA


transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = PaintingDataset(transform=transform)

train_set, val_set, test_set = random_split(dataset, (int(len(dataset)*0.1),
                                                      int(len(dataset)*0.1),
                                                      len(dataset)-int(len(dataset)*0.1)-int(len(dataset)*0.1)))


classes = {'1401-1450', '1451-1500', '1501-1550', '1551-1600',
           '1601-1650', '1651-1700', '1701-1750', '1751-1800', '1801-1850', '1851-1900'}

from torch.utils.data.sampler import SubsetRandomSampler

n_samples = len(dataset)

train_sampler = SubsetRandomSampler(np.arange(len(train_set), dtype=np.int64))
val_sampler = SubsetRandomSampler(np.arange(len(val_set), dtype=np.int64))
test_sampler = SubsetRandomSampler(np.arange(len(test_set), dtype=np.int64))


class SimpleCNN(torch.nn.Module):
    
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.training = True
        self.dropout = 0
        
        #256x256x3
        #Input channels = 3, output channels = 64
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        
        
        self.fc1 = torch.nn.Linear(128 * 64 * 64, 128)        
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        
        x = self.pool1(x)

        x = F.relu(self.conv2(x))

        x = self.pool2(x)
        
        x = x.view(-1, 128 * 64 * 64)
        
        x = F.relu(self.fc1(x))

        x = F.dropout(x, training=self.training, p=self.dropout) if self.training else x

        x = F.relu(self.fc2(x))

        x = F.dropout(x, training=self.training, p=self.dropout) if self.training else x
        
        x = self.fc3(x)

        return(x)

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
def trainNet(net, batch_size, n_epochs, learning_rate, weight_decay):
    
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
    loss, optimizer = createLossAndOptimizer(net, learning_rate, weight_decay)
    #Time for printing
    training_start_time = time.time()
    # f = torch.nn.LogSoftmax(dim=1)
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        net.training = True if net.dropout > 0 else False

        running_loss = 0.0
        print_every = 1#n_batches // 20
        start_time = time.time()
        total_train_loss = 0

        total_tested = 0
        total_correct = 0
        for i, data in enumerate(train_loader):

            #Get inputs
            inputs, labels = data
            one_hot_labels = []
            for label in labels:
                try:
                    one_hot_labels.append(Dict[label])
                except KeyError:
                    continue


            labels = torch.tensor(one_hot_labels)

            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            # print(labels)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize

            outputs = net(inputs)
            
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
            
            #Print every 10th batch of an epoch TODO
            if (i + 1) % (print_every + 1) == 0:
               print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                       epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
               #Reset running loss and time
               running_loss = 0.0
               start_time = time.time()

        print("Accuracy of training set: ", total_correct/total_tested)

        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        total_tested = 0
        total_correct = 0
        net.training = False
        for inputs, labels in val_loader:

            one_hot_labels = []
            for label in labels:
                try:
                    one_hot_labels.append(Dict[label])
                except KeyError:
                    continue
            labels = torch.tensor(one_hot_labels)
            
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            #Forward pass
            val_outputs = net(inputs)
            predicted_labels = torch.tensor([int(x.max(0)[1]) for x in val_outputs])

            true_labels = labels
            for ii in range(len(predicted_labels)):
                total_tested += 1
                if predicted_labels[ii] == true_labels[ii]:
                    total_correct += 1

            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.item()

        print("Accuracy of Test set: ", total_correct/total_tested)
        print("Test loss = {:.2f}".format(total_val_loss / len(val_loader)))
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

case = int(os.environ.get('CASE'))
if case == 0:
    print("\nweight_decay = 0, dropout = 0:\n")
    CNN0 = SimpleCNN()
    CNN0.dropout = 0
    trainNet(CNN0, batch_size=32, n_epochs=10, learning_rate=0.001, weight_decay=0.0) 
    torch.save(CNN0.state_dict(), './v6/CNN0.pth')
elif case == 1:
    print("\nweight_decay = 0, dropout = 0.25\n")
    CNN1 = SimpleCNN()
    CNN1.dropout = .25
    trainNet(CNN1, batch_size=32, n_epochs=10, learning_rate=0.001, weight_decay=0.0) 
    torch.save(CNN1.state_dict(), './v6/CNN1.pth')
elif case == 2:
    print("\nweight_decay = 0, dropout = 0.50\n")
    CNN2 = SimpleCNN()
    CNN2.dropout = .50
    trainNet(CNN2, batch_size=32, n_epochs=10, learning_rate=0.001, weight_decay=0.0)
    torch.save(CNN2.state_dict(), './v6/CNN2.pth')

elif case == 3:
    print("\nweight_decay = 0.05, dropout = 0\n")
    CNN3 = SimpleCNN()
    CNN3.dropout = 0
    trainNet(CNN3, batch_size=32, n_epochs=10, learning_rate=0.001, weight_decay=0.05) 
    torch.save(CNN3.state_dict(), './v6/CNN3.pth')
elif case == 4:
    print("\nweight_decay = 0.05, dropout = 0.25\n")
    CNN4 = SimpleCNN()
    CNN4.dropout = .25
    trainNet(CNN4, batch_size=32, n_epochs=10, learning_rate=0.001, weight_decay=0.05) 
    torch.save(CNN4.state_dict(), './v6/CNN4.pth')
elif case == 5:
    print("\nweight_decay = 0.05, dropout = 0.50\n")
    CNN5 = SimpleCNN()
    CNN5.dropout = .50
    trainNet(CNN5, batch_size=32, n_epochs=10, learning_rate=0.001, weight_decay=0.05)
    torch.save(CNN5.state_dict(), './v6/CNN5.pth')




