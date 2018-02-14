from libCollabAELearn import *

import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import scale

VERBOSE = False
VERBOSE_STEP = 100

# HYPERPARAMETERS
NVIEWS = 3
PTEST = .1

NSTEPS = 100000
NSTEPS_WEIGHTS = 50000

LOSS_METHOD = nn.MSELoss()
# LOSS_METHOD = new_loss

LAYERS_AE = [15]
LAYERS_LINKS = [15, 10]
LAYERS_CLASSIF = [10]

LEARNING_RATE_AE = 0.02
LEARNING_RATE_LINKS = 0.05
LEARNING_RATE_CLASSIF = 0.03
LEARNING_RATE_WEIGHTS = 0.05

LEARN_WEIGHTS = True

MOMENTUM = 0.9
PATIENCE = 200
version = 4
clampOutput = False if version == 4 else True

data = pd.read_csv("data/wdbc/wdbc.data", header=None).values[:,2:]
data = np.array(data, dtype='float')
data = scale(data)

labels = pd.read_csv("data/wdbc/wdbc.data", header=None).values[:,1]
labels, names_labels = labels_as_matrix(labels)

np.random.seed(np.random.randint(10000))
state = np.random.get_state()
np.random.shuffle(data)
np.random.set_state(state)
np.random.shuffle(labels)


# DATA INFORMATIONS
dimData = data.shape[1]
nData = data.shape[0]
nTest = int(nData * PTEST)

# TRAIN AND TEST SETS
test_data = Variable(torch.from_numpy(data[:nTest,:]).float())
train_data = Variable(torch.from_numpy(data[nTest:,:]).float())

test_labels = Variable(torch.LongTensor(labels[:nTest]))
train_labels = Variable(torch.LongTensor(labels[nTest:]))

indexes = getIndexesViews(dimData, NVIEWS)
train_datasets = getViewsFromIndexes(train_data, indexes)
test_datasets = getViewsFromIndexes(test_data, indexes)

print("\n")
print("nData : " + str(nData))
print("Test : " + str(nTest))
print("dim AE : input "+ str(LAYERS_AE))
print("dim Links : input " + str(LAYERS_LINKS + [LAYERS_AE[-1]]))
print("Indexes : " + str(indexes))
print("\n")

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
	learnCollabSystem4(train_datasets, test_datasets, options)

	
