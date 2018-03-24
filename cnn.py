#!/usr/bin/env python3

"""
     Authors: Hung NGUYEN.
     Date: March 2018

     Simple algorithm to classify all the classes of Fashion Mnist dataset.
     The accuracy of this model is ~90%.
"""


import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST, MNIST
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


#########################################################################
# Handle the args to this script
#########################################################################

parser = argparse.ArgumentParser()
parser.add_argument("mode",choices=['train', 'test', 'demo'], help="Different mode of script: train|test|demo")
parser.add_argument("-bs","--batch_size", type = int, default = 128, help="Batch size will used in training phase.")
parser.add_argument("-nbw", "--number_worker", type = int, default = 4, help ="number of worker used on prepare data.")
parser.add_argument("-ep","--epoches", type = int, default=20, help="number of epoches to run.")
parser.add_argument("-mn","--model_name", default='S4L_CNN', help="name of model without extension, the extension is added automatically.")
parser.add_argument("-lr","--learning_rate", type = float, default= 0.01, help="learning rate of training.")
args = parser.parse_args()

#########################################################################
# Prepare for data
#########################################################################

transform = transforms.Compose([transforms.ToTensor(),])  # transforms.Normalize((0.1307,), (0.3081,))])

#########################################################################
# CNN
#########################################################################

class SimpleCNN(nn.Module):
    """
        Simple architecture of CNN
    """
    def __init__(self, nb_channel_in=1, nb_channel_hid=32,nb_channel_hid_2=64, kernel_size=3):
        """
        Args:
            nb_channel_in     : depth (or number of channels) of input image
            nb_channel_hid    : depth of hidden layers
            nb_channel_hid_2  : depth of hidden layers number 2
            kernel_size       : same size of kernel apply for both layers.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(nb_channel_in,nb_channel_hid,kernel_size)
        self.bn_1 = nn.BatchNorm2d(nb_channel_hid)
        self.max_pool_1 = nn.MaxPool2d(2)
        self.max_pool_2 = nn.MaxPool2d(2)
        self.drop_out_1 = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(nb_channel_hid,64,3)
        self.bn_2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(nb_channel_hid_2,128,kernel_size)
        self.bn_3 = nn.BatchNorm2d(128)
        self.drop_out_2 = nn.Dropout(p=0.4)

        sz_at_fc = 3
        nb_classes = 10
        self.fc1 = nn.Linear(sz_at_fc*sz_at_fc * 128, 128)
        self.drop_out_3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128,nb_classes)
        self.sm = nn.Softmax(dim = 1)

    def forward(self,x):
        """
            Args:
                x: batch of image size of (num_images = 64,c = 1, h = 28, w = 28)
        """
        hl1 = F.relu(self.bn_1(self.conv1(x)))
        hl1 = self.max_pool_1(hl1)
        hl1 = self.drop_out_1(hl1)
        hl2 = F.relu(self.bn_2(self.conv2(hl1)))
        hl2 = self.max_pool_2(hl2)
        hl2 = self.drop_out_1(hl2)
        hl3 = F.relu(self.bn_3(self.conv3(hl2)))
        hl3 = self.drop_out_2(hl3)

        # Flat the each of sample inside batch.
        hl3 = hl3.view(hl3.size()[0], -1)
        hl4 = self.fc1(hl3)
        hl4 = self.drop_out_3(hl4)
        hl5 = self.fc2(hl4)
        return self.sm(hl5)

#########################################################################
# Name of model file
#########################################################################

model_name = args.model_name + '.pt'

#########################################################################
# Training
#########################################################################
if args.mode=='train':
    #load dataset to loader

    dataset = FashionMNIST("./data", train=True, download = True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.number_worker)


    print(len(dataset))
    net = SimpleCNN().cuda()

    # number of epochs
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.xavier_uniform(m.weight)

        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    # initialize the weights in CNN.
    net.apply(weights_init)

    criterion = nn.CrossEntropyLoss()
    print("Learning rate:", args.learning_rate)

    # Trying with different optimizer.
    #opt = optim.SGD(net.parameters(), lr=args.learning_rate, momentum = 0.9)
    opt = optim.Adam(net.parameters(), lr=args.learning_rate)

    # store losses so we could draw it later
    all_losses = []

    for ep in range(args.epoches):
        current_loss = 0.0

        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            opt.zero_grad()

            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            opt.step()

            current_loss += loss.data[0]
            if i % 10 == 0:
                current_loss /= 10.0
                all_losses.append(current_loss)
                print("Info: ", str(ep), str(i), str(current_loss))
                current_loss = 0.0

    ##########################################################################
    # training now finish, save the model and draw losses
    ##########################################################################
    torch.save(net,model_name)
    plt.plot(all_losses)
    plt.savefig('loss.png')

#############################################################################
# Test mode
############################################################################

if args.mode=='test':
    net = torch.load(model_name)
    dataset = FashionMNIST("./data", train=False, download = True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size, shuffle=True, num_workers=args.number_worker)
    print(len(dataset))
    net.eval()
    test_loss = 0
    correct = 0
    loss_func = nn.CrossEntropyLoss(size_average = False)
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += loss_func(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
                                                                                    100. * correct / len(test_loader.dataset)))





