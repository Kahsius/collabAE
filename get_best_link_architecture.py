#!/home/denis/python/collabAE/bin/python3
from os import listdir
from torch.autograd import Variable
from sklearn.preprocessing import scale
from libCollabAELearn import *

import torch.nn as nn
import pandas as pd
import torch
import numpy as np
import pdb

archis = [[100], [200], [300], [400], [500], [600], [700],  [100, 100], [200, 200], [300, 300], [400, 400], [500, 500], [600, 600], [700, 700]]
NVIEWS = 4
VERBOSE = False
VERBOSE_STEP = 100
BIG_TEST = False
BIG_TEST_ITER = 50

# NOISE TYPE
NOISY_ON_DATA = False
NOISY_VIEW = False
NOISY_VIEW_0 = False

# HYPERPARAMETERS
PTEST = .1
NSTEPS = 5000
LOSS_METHOD = nn.MSELoss()
LEARNING_RATE_LINKS = 0.05
MOMENTUM = 0.9
PATIENCE = 200
version = 4
clampOutput = False if version == 4 else True

base = "data/madelon/"
files = listdir(base)
files = [name for name in files if 'code' in name]
files.sort()
codes = [Variable(torch.FloatTensor(pd.read_csv(base + name).values), requires_grad = False) for name in files]

files = listdir(base)
files = [name for name in files if 'codtest' in name]
files.sort()
codes_test = [Variable(torch.FloatTensor(pd.read_csv(base + name).values), requires_grad = False) for name in files]

data = pd.read_csv("data/madelon/madelon_train.data", sep=" ", header=None).values[:,:-1]
data = np.array(data, dtype='float')
data = scale(data)

data_test = pd.read_csv("data/madelon/madelon_valid.data", sep=" ", header=None).values[:,:-1]
data_test = np.array(data_test, dtype='float')
data_test = scale(data_test)

# DATA INFORMATIONS
dimData = data.shape[1]
nData = data.shape[0]
nTest = int(nData * PTEST)

# TRAIN AND TEST SETS
test_data = Variable(torch.from_numpy(data_test).float())
train_data = Variable(torch.from_numpy(data).float())

indexes = getIndexesViews(dimData, NVIEWS)
train_datasets = getViewsFromIndexes(train_data, indexes)
test_datasets = getViewsFromIndexes(test_data, indexes)

r = []
for archi in archis :
    NVIEWS = len(codes)
    LAYERS_LINKS = archi
    print("Learn archi " + str(archi))

    options = {
    	"VERBOSE"               : VERBOSE,
    	"VERBOSE_STEP"          : VERBOSE_STEP,
    	"NSTEPS"                : NSTEPS,
    	"LAYERS_LINKS"          : LAYERS_LINKS,
    	"LEARNING_RATE_LINKS"   : LEARNING_RATE_LINKS,
    	"MOMENTUM"              : MOMENTUM,
    	"PATIENCE"              : PATIENCE,
    	"LOSS_METHOD"           : LOSS_METHOD,
    	"clampOutput"           : clampOutput,
    	"version"               : version,
        "BASE"                  : "madelon"
    }
    
    dic = {
        "id_in" : 0,
        "id_out" : 1,
        "data_in" : codes[0],
        "data_out" : train_datasets[1],
        "test_in" : codes_test[0],
        "test_out" : test_datasets[1],
        "options" : options,
    }

    model, result = learn_LinkNet(dic)
    r.append(result)
    
print(r)
