from libCollabAELearn import *

import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import scale
from sys import exit
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

NVIEWS = 3

VERBOSE = True
VERBOSE_STEP = 100

# HYPERPARAMETERS
PTEST = .1

NSTEPS = 5000
NSTEPS_WEIGHTS = 1000

LAYERS_AE = [150, 50, 20]
LAYERS_LINKS = [50, 100]
LAYERS_CLASSIF = [150]

LEARNING_RATE_AE = 0.03
LEARNING_RATE_LINKS = 0.05
LEARNING_RATE_CLASSIF = 0.05
LEARNING_RATE_WEIGHTS = 0.01

MOMENTUM = 0.9
PATIENCE = 200

version = 4
clampOutput = False if version == 4 else True

LOSS_METHOD = nn.MSELoss()

# GET DATA
#Dataset
train_dataset = MNIST(root = "data/mnist/", transform = ToTensor())
test_dataset = MNIST(root = "data/mnist/", transform = ToTensor(), train = False)

dim = train_dataset.train_data.shape
dimData = ft.reduce(lambda x, y : x*y, dim[1:])
train_dataset.train_data.resize_(dim[0], dimData)
dim = test_dataset.test_data.shape
test_dataset.test_data.resize_(dim[0], dimData)

# Labels
train_labels = Variable(train_dataset.train_labels)
test_labels = Variable(test_dataset.test_labels)

train_dataset = Variable(train_dataset.train_data.float())
print("Train dataset size : " + str(train_dataset.shape))
test_dataset = Variable(test_dataset.test_data.float())
print("Test dataset size : " + str(test_dataset.shape))

indexes = getIndexesViews(dimData, NVIEWS)
train_datasets = getViewsFromIndexes(train_dataset, indexes)
test_datasets = getViewsFromIndexes(test_dataset, indexes)

# print("\n")
# print("nData : " + )
# print("Test : " + str(nTest))
# print("dim AE : input "+ str(LAYERS_AE))
# print("dim Links : input " + str(LAYERS_LINKS + [LAYERS_AE[-1]]))
# print("Indexes : " + str(indexes))
# print("\n")

options = {
    "VERBOSE" : VERBOSE,
    "VERBOSE_STEP" : VERBOSE_STEP,
    "NSTEPS" : NSTEPS,
    "NSTEPS_WEIGHTS" : NSTEPS_WEIGHTS,
    "LAYERS_AE" : LAYERS_AE,
    "LAYERS_LINKS" : LAYERS_LINKS,
    "LAYERS_CLASSIF" : LAYERS_CLASSIF,
    "LEARNING_RATE_AE" : LEARNING_RATE_AE,
    "LEARNING_RATE_LINKS" : LEARNING_RATE_WEIGHTS,
    "LEARNING_RATE_WEIGHTS" : LEARNING_RATE_WEIGHTS,
    "LEARNING_RATE_CLASSIF" : LEARNING_RATE_CLASSIF,
    "MOMENTUM" : MOMENTUM,
    "PATIENCE" : PATIENCE,
    "LEARN_WEIGHTS" : True,
    "LOSS_METHOD" : LOSS_METHOD,
    "train_labels" : train_labels,
    "test_labels" : test_labels,
    "clampOutput" : clampOutput,
    "version" : version
}

if version == 3 :
    learnCollabSystem3(train_datasets, test_datasets, options)
elif version == 4 or version == 5:
    learnCollabSystem4(train_datasets, test_datasets, options)

    
