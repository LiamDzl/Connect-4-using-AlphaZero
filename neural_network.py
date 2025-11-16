import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from connect_4 import mask

class policy(nn.Module):
    def __init__(self, structure): 
        super().__init__() # because of inheritance
        layers = [] # this object is a python list, but contains actual affine maps 

        for i in range(len(structure) - 1):  # n layers need n-1 Linear maps
            layers.append(nn.Linear(structure[i], structure[i + 1]))

        self.structure = structure
        self.layers = nn.ModuleList(layers)
        self.pspace_dimensions = []

        for i in range(len(structure) - 1):
            self.pspace_dimensions.append(structure[i] * structure[i+1]) # size of weight matrix
            self.pspace_dimensions.append(structure[i+1]) # size of bias vector

        self.total_dimension = 0
        for i in self.pspace_dimensions:
            self.total_dimension += i
     
    def forward(self, state): # mask a list of columns that are full, i.e. top element non-zero - btw mask comes automatically from state
        x = state.reshape(1,42)
        for i in self.layers[:-1]: # ReLU, up til penultimate (since we wanna grab this vector)
            x = F.relu(i(x))

        penultimate_eight = self.layers[-1](x)
        value = penultimate_eight[:,-1]
        filter = mask(state)
        to_softmax = penultimate_eight[filter] #<=7
        print(to_softmax)

        reduced_distribution = F.softmax(to_softmax, dim = 0)
        print(reduced_distribution)
        distribution = torch.zeros(7,1)
        count = 0
        for i in range(7):
            if filter[0, i] == True:
                distribution[i] = reduced_distribution[count]
                count += 1
        
        return value, distribution