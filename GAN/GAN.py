import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd
from skimage import io, transform

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets, utils

from torch.utils.data import Dataset, DataLoader

from __future__ import print_function, division

from google.colab import files

import time
torch.backends.cudnn.deterministic=True

import glob

import warnings
warnings.filterwarnings("ignore")

plt.ion()  

from google.colab import drive

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_set = torchvision.datasets.MNIST(root = '/content/gdrive/My Drive/Colab Notebooks/GAN', download = True, transform = transform)
train_loader = DataLoader(data_set, batch_size = 128, shuffle = True, num_workers = 1)

z_size = 100
output_size = 784

class Generator(nn.Module):
  '''Generates an image from gaussian noise'''
  
  def __init__(self, z_size, hidden_size, hidden_size_2, hidden_size_3, output_size):
    super(Generator, self).__init__()
    self.fc1 = nn.Linear(z_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size_2)
    self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
    self.fc4 = nn.Linear(hidden_size_3, output_size)
    self.leakyrelu = nn.LeakyReLU(0.3)
    self.tanh = nn.Tanh()
  
  def forward(self, z):
    x = self.leakyrelu(self.fc1(z))
    x = self.leakyrelu(self.fc2(x))
    x = self.leakyrelu(self.fc3(x))
    x = self.tanh(self.fc4(x))
    return x

class Discriminator(nn.Module):
  '''Discriminates between real samples and fake samples'''
  
  def __init__(self, image_size, hidden_size_1, hidden_size_2, hidden_size_3):
    super(Discriminator, self).__init__()
    self.fc1 = nn.Linear(image_size, hidden_size_1)
    self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
    self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
    self.fc4 = nn.Linear(hidden_size_3, 1)
    self.leakyrelu = nn.LeakyReLU(0.4) #By experience, when using a too small leaky relu coefficient, we get less diversity of samples
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(0.2)
    
  def forward(self, x):
    y = self.leakyrelu(self.fc1(x))
    y = self.leakyrelu(self.fc2(y))
    y = self.dropout(self.leakyrelu(self.fc3(y)))
    y = self.sigmoid(self.fc4(y))
    return y

class Train_model():
  '''Trains both discriminator and generator'''
  
  def __init__(self, generator, discriminator, k, data_loader, batch_size, z_size):
    self.generator = generator
    self.discriminator = discriminator
    self.k = k
    self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr = 0.0001)
    self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr = 0.0001)
    self.data_loader = data_loader
    self.batch_size = batch_size
    self.z_size = z_size
    self.nb_epochs = 200
    self.criterion = nn.BCELoss()
    
  def label_smoothing(self, value, data_size):
    label = torch.zeros(data_size)
    if value == 0: label.uniform_(0, 0.1)
    else : label.uniform_(0.9, 1)
    return label
          
  def train(self):
    
    losses = []
    
    for epoch in range(self.nb_epochs):
      
      
      #self.true_labels = torch.zeros()      
      #z_noise = torch.randn(1, 100)
      #print(generator(z_noise))
    
      for i, data in enumerate(self.data_loader, 0):

        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        
        size_0, size_1, size_2, size_3 = data[0].size()
        data = data[0].view(size_0, 784).cuda()

        for j in range(self.k):

          z_noise = torch.randn(size_0, self.z_size).cuda()
          generated = self.generator(z_noise)
          
          true_labels = self.label_smoothing(1, size_0).cuda()
          fake_labels = self.label_smoothing(0, size_0).cuda()

          d_output_data = self.discriminator(data)
          d_output_generated = self.discriminator(generated)
          d_loss = self.criterion(d_output_data, true_labels) + self.criterion(d_output_generated, fake_labels)
          d_loss.backward()
          self.d_optimizer.step()

        z_noise = torch.randn(size_0, self.z_size).cuda()
        generated = self.generator(z_noise)
        d_output_generated = self.discriminator(generated)
        fake_labels = self.label_smoothing(1, size_0).cuda()

        g_loss = self.criterion(d_output_generated, fake_labels)
        g_loss.backward()
        
        if i == 0 : 
          losses.append((d_loss.data.cpu().numpy(), g_loss.data.cpu().numpy()))
        
        self.g_optimizer.step()

      if epoch % 5 == 0: print("Epoch", epoch, "D_loss :", losses[-1][0], "G_loss :", losses[-1][1])

generator = Generator(100, 200, 400, 800, 784).cuda()
discriminator = Discriminator(784, 800, 400, 200).cuda()

train_model = Train_model(generator, discriminator, 1, train_loader, 128, 100)

train_model.train()

fig = plt.figure(figsize=(10, 10))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    z_noise = torch.randn(1, 100).cuda()
    generated = generator(z_noise)
    fig.add_subplot(rows, columns, i)
    plt.imshow(generated.view(28, 28).cpu().detach().numpy())
plt.show()
