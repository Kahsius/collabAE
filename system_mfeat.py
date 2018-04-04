from libCollabAELearn import *

import pandas as pd
import numpy as np
import pickle
import time

from sklearn.preprocessing import scale

VERBOSE = True
VERBOSE_STEP = 1
BIG_TEST = False
BIG_TEST_ITER = 20

# HYPERPARAMETERS
NSTEPS = 10000
NSTEPS_WEIGHTS = 2

NOISY_ON_DATA = True

LAYERS_AE = [150]
LAYERS_LINKS = [150]
LAYERS_CLASSIF = [150]

LEARNING_RATE_AE = 0.15
LEARNING_RATE_LINKS = 0.05
LEARNING_RATE_CLASSIF = 0.05
LEARNING_RATE_WEIGHTS = 0.01

MOMENTUM = 0.9
PATIENCE = 200

LEARN_WEIGHTS = True

version = 4
clampOutput = False if version == 4 else True

LOSS_METHOD = nn.MSELoss()

# GET DATA
state = np.random.get_state()

# Labels
labels = []
for i in range(10) :
    labels += [i] * 200
np.random.set_state(state)
np.random.shuffle(labels)
labels, _ = labels_as_matrix(labels)
train_labels = Variable(torch.LongTensor(labels[:-200]))
test_labels = Variable(torch.LongTensor(labels[-200:]))

# Datasets
train_datasets = []
test_datasets = []
for file in ["fac", "fou", "kar", "mor", "pix", "zer"] :
    data = pd.read_csv("data/mfeat/mfeat-" + file, header=None, delim_whitespace=True)
    data = np.array(data, dtype='float')
    data = scale(data)
    if NOISY_ON_DATA :
        data += np.random.normal(size=data.shape, scale=1)
    np.random.set_state(state)
    np.random.shuffle(data)
    train_datasets.append(Variable(torch.from_numpy(data[:-200,:]).float()))
    test_datasets.append(Variable(torch.from_numpy(data[-200:,:]).float()))

NVIEWS = len(train_datasets)

# print("\n")
# print("nData : " + )
# print("Test : " + str(nTest))
# print("dim AE : input "+ str(LAYERS_AE))
# print("dim Links : input " + str(LAYERS_LINKS + [LAYERS_AE[-1]]))
# print("Indexes : " + str(indexes))
# print("\n")

options = {
    "VERBOSE"               : VERBOSE,
    "VERBOSE_STEP"          : VERBOSE_STEP,
    "NSTEPS"                : NSTEPS,
    "NSTEPS_WEIGHTS"        : NSTEPS_WEIGHTS,
    "LAYERS_AE"             : LAYERS_AE,
    "LAYERS_LINKS"          : LAYERS_LINKS,
    "LAYERS_CLASSIF"        : LAYERS_CLASSIF,
    "LEARNING_RATE_AE"      : LEARNING_RATE_AE,
    "LEARNING_RATE_LINKS"   : LEARNING_RATE_WEIGHTS,
    "LEARNING_RATE_WEIGHTS" : LEARNING_RATE_WEIGHTS,
    "LEARNING_RATE_CLASSIF" : LEARNING_RATE_CLASSIF,
    "MOMENTUM"              : MOMENTUM,
    "PATIENCE"              : PATIENCE,
    "LEARN_WEIGHTS"         : LEARN_WEIGHTS,
    "LOSS_METHOD"           : LOSS_METHOD,
    "train_labels"          : train_labels,
    "test_labels"           : test_labels,
    "clampOutput"           : clampOutput,
    "nLabels"               : 10,
    "version"               : version
}

if version == 3 :
    learnCollabSystem3(train_datasets, test_datasets, options)
elif version == 4 or version == 5:
    if BIG_TEST :
        for i in range(BIG_TEST_ITER) :
            print("Test " + str(i))
            print("Begin time : " + str(time.localtime()))
            results = learnCollabSystem4(train_datasets, test_datasets, options)
            f = "data/mfeat/results_standard_noisy/results_" + str(i)
            f = open(f, "w+b")
            pickle.dump(results, f)
            f.close()
            print("End time : " + str(time.localtime()))
    else :
        learnCollabSystem4(train_datasets, test_datasets, options)
