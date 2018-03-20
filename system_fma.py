from libCollabAELearn import *

import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np

from sklearn.preprocessing import scale

# OPTIONS
VERBOSE = True
VERBOSE_STEP = 100
BIG_TEST = False
BIG_TEST_ITER = 20

# HYPERPARAMETERS
NSTEPS = 1000
NSTEPS_WEIGHTS = 1000
CHUNKSIZE = 100

NOISY_ON_DATA = False

LAYERS_AE = [50, 10]
LAYERS_LINKS = [50]
LAYERS_CLASSIF = [50]

LEARNING_RATE_AE = 0.04
LEARNING_RATE_LINKS = 0.05
LEARNING_RATE_CLASSIF = 0.05
LEARNING_RATE_WEIGHTS = 0.01

MOMENTUM = 0.9
PATIENCE = 100

LEARN_WEIGHTS = True

version = 4
clampOutput = False if version == 4 else True

LOSS_METHOD = nn.MSELoss()

# GET DATA
print("{0} tracks described by {1} features".format(*features.shape))
columns = ['mfcc', 'chroma_cens', 'tonnetz', 'spectral_contrast', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'rmse', 'zcr']

train_datasets = []
test_datasets = []

for column in columns :
    data = pd.read_csv("data/fma/"+column+"_train.csv",
        chunksize=CHUNKSIZE)
    data = map(lambda chunk: Variable(chunk.values))
    train_datasets.append(data)

    data = pd.read_csv("data/fma/"+column"_test_csv")
    test_datasets.append(data)

NVIEWS = len(columns)

train_labels = pd.read_csv("data/fma/labels_train.csv",
    chunksize=CHUNKSIZE)
test_labels = pd.read_csv("data/fma/labels_test.csv")

options = {
    "VERBOSE" : VERBOSE,
    "VERBOSE_STEP" : VERBOSE_STEP,
    "NSTEPS" : NSTEPS,
    "NSTEPS_WEIGHTS" : NSTEPS_WEIGHTS,
    "LAYERS_AE" : LAYERS_AE,
    "LAYERS_LINKS" : LAYERS_LINKS,
    "LAYERS_CLASSIF" : LAYERS_CLASSIF,
    "LEARNING_RATE_AE" : LEARNING_RATE_AE,
    "LEARNING_RATE_LINKS" : LEARNING_RATE_LINKS,
    "LEARNING_RATE_WEIGHTS" : LEARNING_RATE_WEIGHTS,
    "LEARNING_RATE_CLASSIF" : LEARNING_RATE_CLASSIF,
    "MOMENTUM" : MOMENTUM,
    "PATIENCE" : PATIENCE,
    "LEARN_WEIGHTS" : LEARN_WEIGHTS,
    "LOSS_METHOD" : LOSS_METHOD,
    "train_labels" : train_labels,
    "test_labels" : test_labels,
    "clampOutput" : clampOutput,
    "version" : version
}

if version == 3 :
    learnCollabSystem3(train_datasets, test_datasets, options)
elif version == 4 or version == 5:
    if BIG_TEST :
        for i in range(BIG_TEST_ITER) :
            print("Test " + str(i))
            print("Begin time : " + str(time.localtime()))
            results = learnCollabSystem4(train_datasets, test_datasets, options)
            f = "data/fma/results_standard/results_" + str(i)
            f = open(f, "w+b")
            pickle.dump(results, f)
            f.close()
            print("End time : " + str(time.localtime()))
    else :
        learnCollabSystem4(train_datasets, test_datasets, options)