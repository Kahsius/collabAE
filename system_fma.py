import pdb
import time
import pickle

from sklearn.preprocessing import scale
import torch
from torch.autograd import Variable

from libCollabAELearn import *
import pandas as pd


# OPTIONS
VERBOSE = True
VERBOSE_STEP = 100
BIG_TEST = False
BIG_TEST_ITER = 20

# HYPERPARAMETERS
NSTEPS = 500
NSTEPS_WEIGHTS = 500
CHUNKSIZE = 1000

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
columns = ['mfcc', 'chroma_cens', 'tonnetz', 'spectral_contrast', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'rmse', 'zcr']

train_datasets = []
test_datasets = []

for column in columns :
#    data = pd.read_csv("data/fma/"+column+"_train.csv",
#        chunksize=CHUNKSIZE)
#    data = map(lambda chunk: Variable(torch.from_numpy(chunk.values).float()), data)
    data = "data/fma/"+column+"_train.csv"
    train_datasets.append(data)

    data = pd.read_csv("data/fma/"+column+"_test.csv")
    test_datasets.append(Variable(torch.from_numpy(data.values).float()))

NVIEWS = len(columns)

train_labels = "data/fma/labels_train.csv"
#train_labels = pd.read_csv("data/fma/labels_train.csv",
#    chunksize=CHUNKSIZE)
#train_labels = map(lambda chunk: Variable(torch.from_numpy(chunk.values).squeeze().long()), train_labels)
test_labels = Variable(torch.from_numpy(pd.read_csv("data/fma/labels_test.csv").values).squeeze().long())

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
    "version" : version,
    "nLabels" : 16,
    "CHUNKSIZE": CHUNKSIZE
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
