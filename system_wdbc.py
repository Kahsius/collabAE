from collabAE import *
import pandas as pd
import numpy as np

VERBOSE = False
VERBOSE_STEP = 100

# HYPERPARAMETERS
PTEST = .1
NSTEPS = 200
NSTEPS_WEIGHTS = 200
NVIEWS = 3
LAYERS_AE = [10]
LAYERS_LINKS = [20]
LEARNING_RATE_AE = 0.03
LEARNING_RATE_LINKS = 0.06
LEARNING_RATE_WEIGHTS = 0.01
MOMENTUM = 0.9
PATIENCE = 100

data = pd.read_csv("data/wdbc.data", header=None).values[:,2:]
data = np.array(data, dtype='float')
data = scale(data)
np.random.shuffle(data)

# DATA INFORMATIONS
dimData = data.shape[1]
nData = data.shape[0]
nTest = int(nData * PTEST)

# TRAIN AND TEST SETS
test_data = Variable(torch.from_numpy(data[:nTest,:]).float())
train_data = Variable(torch.from_numpy(data[nTest:,:]).float())

indexes = getIndexesViews(dimData, NVIEWS)
train_datasets = getViewsFromIndexes(train_data, indexes)
test_datasets = getViewsFromIndexes(test_data, indexes)

print("\n")
print("nData : " + str(nData))
print("Test : " + str(nTest))
print("dim AE : " + str(["input"] + LAYERS_AE))
print("dim Links : " + str([LAYERS_AE[-1]] + LAYERS_LINKS + [LAYERS_AE[-1]]))
print("Indexes : " + str(indexes))
print("\n")

options = {
	"VERBOSE" : VERBOSE,
	"VERBOSE_STEP" : VERBOSE_STEP,
	"NSTEPS" : NSTEPS,
	"NSTEPS_WEIGHTS" : NSTEPS_WEIGHTS,
	"LAYERS_AE" : LAYERS_AE,
	"LAYERS_LINKS" : LAYERS_LINKS,
	"LEARNING_RATE_AE" : LEARNING_RATE_AE,
	"LEARNING_RATE_LINKS" : LEARNING_RATE_WEIGHTS,
	"LEARNING_RATE_WEIGHTS" : LEARNING_RATE_WEIGHTS,
	"MOMENTUM" : MOMENTUM,
	"PATIENCE" : PATIENCE
}

learnCollabSystem(train_datasets, test_datasets, options)