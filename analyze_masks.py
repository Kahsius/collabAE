#!/home/denis/python/collabAE/bin/python

import pdb
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from results_wdbc import get_results

path = "data/cube/results_standard/"
list_results = os.listdir(path)
K = len(list_results)

data = pickle.load(open(path+list(list_results)[0], "r+b"))
d = data[0]
NVIEWS = len(d["error_classifiers_test"])


masks = []
masks_sd = []
for i in range(3):
    masks.append(data[1].w[i].detach().numpy()/K)

for n, filename in enumerate(list_results) :
    if n>0:
        f = open(path+filename, "r+b")
        data = pickle.load(f)[1]
        m = data.w
        for i in range(NVIEWS) :
            masks[i] += m[i].detach().numpy()/K

print(masks)

data = pickle.load(open(path+list(list_results)[0], "r+b"))

for i in range(3):
    masks_sd.append(np.power(data[1].w[i].detach().numpy()-masks[i], 2)/K)

for filename in list_results:
    f = open(path+filename, "r+b")
    data=pickle.load(f)[1]
    m = data.w
    for i in range(NVIEWS) :
        masks_sd[i] += np.power(m[i].detach().numpy()-masks[i], 2)/K

print(np.sqrt(masks_sd))
