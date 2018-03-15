from libCollabAELearn import *

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
path = "data/fma/fma_metadata/"
features = pandas.read_csv(path + "features.csv")
echonest = pandas.read_csv(path + "echonest.csv")

print("{0} tracks described by {1} features".format(*features.shape))

columns = ['mfcc', 'chroma_cens', 'tonnetz', 'spectral_contrast']
columns.append(['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff'])
columns.append(['rmse', 'zcr'])

for column in columns :
    