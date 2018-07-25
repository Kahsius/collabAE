#!/home/denis/python/collabAE/bin/python

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pickle
import time
import seaborn as sb
import matplotlib.pylab as plt
from libCollabAELearn import *

import pandas as pd
import numpy as np

from sklearn.preprocessing import scale

VERBOSE = False
VERBOSE_STEP = 100
BIG_TEST = True
BIG_TEST_ITER = 200

# HYPERPARAMETERS
NSTEPS = 3000
NSTEPS_WEIGHTS = 3000

NOISY_ON_DATA = True 

LAYERS_AE = [5]
LAYERS_LINKS = [20]

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

labels = []
train_datasets = []
test_datasets = []
for i in range(4):
    labels += [i] * 250
np.random.set_state(state)
np.random.shuffle(labels)
labels, _ = labels_as_matrix(labels)
train_labels = Variable(torch.LongTensor(labels[:-100]))
test_labels = Variable(torch.LongTensor(labels[-100:]))

centers = [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
data = np.zeros((1000,3))
for i in range(4):
    d = np.tile(centers[i], (250, 1))
    d += np.random.normal(size=d.shape, scale=.1)
    data[250*i:250*(i+1),:] = d

# Plot 3D
# colors = ['#505050']*250 +['6E6E6E']*250 +['#000000']*250 + ['#A0A0A0']*250
# colors, _ = labels_as_matrix(colors)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xs=data[:,0], ys=data[:,1], zs=data[:,2], c=colors)
# plt.show()
# pdb.set_trace()

np.random.set_state(state)
np.random.shuffle(data)
perms = ((0,1),(0,2),(1,2))
for i in range(3):
    dataset = data[:,perms[i]]
    train_datasets.append(Variable(torch.from_numpy(dataset[:-100,:]).float()))
    test_datasets.append(Variable(torch.from_numpy(dataset[-100:,:]).float()))

NVIEWS = len(train_datasets)

options = {
    "BASE"                  : "mfeat",
    "VERBOSE"               : VERBOSE,
    "VERBOSE_STEP"          : VERBOSE_STEP,
    "NSTEPS"                : NSTEPS,
    "NSTEPS_WEIGHTS"        : NSTEPS_WEIGHTS,
    "LAYERS_AE"             : LAYERS_AE,
    "LAYERS_LINKS"          : LAYERS_LINKS,
    "LEARNING_RATE_AE"      : LEARNING_RATE_AE,
    "LEARNING_RATE_LINKS"   : LEARNING_RATE_WEIGHTS,
    "LEARNING_RATE_WEIGHTS" : LEARNING_RATE_WEIGHTS,
    "LEARNING_RATE_CLASSIF" : LEARNING_RATE_CLASSIF,
    "MOMENTUM"              : MOMENTUM,
    "PATIENCE"              : PATIENCE,
    "LEARN_WEIGHTS"         : LEARN_WEIGHTS,
    "LOSS_METHOD"           : LOSS_METHOD,
    "PRINT_WEIGHTS"         : False,
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
            results = learnCollabSystem4(train_datasets, test_datasets, options)
            f = "data/cube/"
            f = f + "results_standard/" if LEARN_WEIGHTS else f + "results_sans_weights/"
            f += "results_" + str(i)
            f = open(f, "w+b")
            pickle.dump(results, f)
            f.close()
    else :
        results, cs = learnCollabSystem4(train_datasets, test_datasets, options)
        # images = cs.forward(4, test_datasets).data.numpy()
