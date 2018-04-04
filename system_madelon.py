#!/home/denis/python/collabAE/bin/python3

from libCollabAELearn import *

import pdb
import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.preprocessing import scale

VERBOSE = True
VERBOSE_STEP = 100
BIG_TEST = False
BIG_TEST_ITER = 50
PRINT_WEIGHTS = False

# NOISE TYPE
NOISY_ON_DATA = False
NOISY_VIEW = False
NOISY_VIEW_0 = False

# HYPERPARAMETERS
NVIEWS = 4
PTEST = .1
NSTEPS = 1000
NSTEPS_WEIGHTS = 2000
LOSS_METHOD = nn.MSELoss()
LAYERS_AE = [100]
LAYERS_LINKS = [100, 100]
LEARNING_RATE_AE = 0.05
LEARNING_RATE_LINKS = 0.05
LEARNING_RATE_WEIGHTS = 0.05
LEARN_WEIGHTS = False
MOMENTUM = 0.9
PATIENCE = 200
version = 4
clampOutput = False if version == 4 else True

data = pd.read_csv(
    "data/madelon/madelon_train.data", sep=" ", header=None).values[:, :-1]
data = np.array(data, dtype='float')
data = scale(data)

data_test = pd.read_csv(
    "data/madelon/madelon_valid.data", sep=" ", header=None).values[:, :-1]
data_test = np.array(data_test, dtype='float')
data_test = scale(data_test)

labels = pd.read_csv(
    "./data/madelon/madelon_train.labels", header=None, squeeze=True).values
labels_test = pd.read_csv(
    "./data/madelon/madelon_valid.labels", header=None, squeeze=True).values

# DATA INFORMATIONS
dimData = data.shape[1]
nData = data.shape[0]
nTest = int(nData * PTEST)

# TRAIN AND TEST SETS
test_data = Variable(torch.from_numpy(data_test).float())
train_data = Variable(torch.from_numpy(data).float())

test_labels = Variable(torch.LongTensor(labels_test))
train_labels = Variable(torch.LongTensor(labels))

indexes = getIndexesViews(dimData, NVIEWS)
train_datasets = getViewsFromIndexes(train_data, indexes)
test_datasets = getViewsFromIndexes(test_data, indexes)

if NOISY_VIEW:
    train_datasets.append(
        Variable(
            torch.from_numpy(np.random.normal(size=(nData - nTest,
                                                    10))).float()))
    test_datasets.append(
        Variable(torch.from_numpy(np.random.normal(size=(nTest, 10))).float()))

if NOISY_VIEW_0:
    data = train_datasets[0].data.numpy()
    train_datasets[0] = Variable(
        torch.from_numpy(data + np.random.normal(size=data.shape)).float())
    data = test_datasets[0].data.numpy()
    test_datasets[0] = Variable(
        torch.from_numpy(data + np.random.normal(size=data.shape)).float())

print("\n")
print("nData : " + str(nData))
print("Test : " + str(nTest))
print("dim AE : input " + str(LAYERS_AE))
print("dim Links : input " + str(LAYERS_LINKS + [LAYERS_AE[-1]]))
print("Indexes : " + str(indexes))
print("\n")

options = {
    "VERBOSE"               : VERBOSE,
    "VERBOSE_STEP"          : VERBOSE_STEP,
    "NSTEPS"                : NSTEPS,
    "NSTEPS_WEIGHTS"        : NSTEPS_WEIGHTS,
    "LAYERS_AE"             : LAYERS_AE,
    "LAYERS_LINKS"          : LAYERS_LINKS,
    "LEARNING_RATE_AE"      : LEARNING_RATE_AE,
    "LEARNING_RATE_LINKS"   : LEARNING_RATE_WEIGHTS,
    "LEARNING_RATE_WEIGHTS" : LEARNING_RATE_WEIGHTS,
    "MOMENTUM"              : MOMENTUM,
    "PATIENCE"              : PATIENCE,
    "LEARN_WEIGHTS"         : LEARN_WEIGHTS,
    "LOSS_METHOD"           : LOSS_METHOD,
    "train_labels"          : train_labels,
    "test_labels"           : test_labels,
    "clampOutput"           : clampOutput,
    "nLabels"               : 2,
    "version"               : version,
    "PRINT_WEIGHTS"         : PRINT_WEIGHTS
}
if version == 3:
    learnCollabSystem3(train_datasets, test_datasets, options)
elif version == 4 or version == 5:
    results = learnCollabSystem4(train_datasets, test_datasets, options)
    f = open("data/madelon/results_sans_weights/results", "w+b")
    pickle.dump(results, f)
    # if BIG_TEST:
#         for i in range(BIG_TEST_ITER):
            # results = learnCollabSystem4(train_datasets, test_datasets,
                                         # options)
            # f = "data/wdbc/results_sans_weights/results_" + str(i)
            # f = open(f, "w+b")
            # pickle.dump(results, f)
            # W.close()
