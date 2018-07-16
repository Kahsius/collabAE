import pdb
import sys
import pandas as pd
import torch
import functools as ft
from torch.autograd import Variable
from numpy import asarray


def getIndexesViews(dimData, nViews):
    indexes = list()
    for i in range(nViews+1):
        indexes.append(round(i*dimData/nViews))
    return indexes

# =====================================================================

def getViewsFromIndexes(data, indexes):
    views = list()
    for i in range(len(indexes)-1):
        views.append(data[:,indexes[i]:indexes[i+1]])
    return views

# =====================================================================

def getWeightedInputCodes(i, codes, links, weights):
    w_codes = list()
    for j in range(len(codes)):
        if i != j :
            code_externe = codes[j]
            code_interne = links[j][i](code_externe)*weights[j]
            w_codes.append(code_interne)

    code_moyen = ft.reduce(lambda x, y: x+y, w_codes)

    return code_moyen

# =====================================================================

def get_args_to_map_AE(train_datasets, test_datasets, options):
    f = lambda i : { "train" : train_datasets[i],"test" : test_datasets[i], "options" : options, "id_net" : i}
    return map(f, range(len(train_datasets)))

# =====================================================================

def get_args_to_map_weights(train_datasets, test_datasets, models, links, codes, codes_test, options):
    NVIEWS = len(train_datasets)
    args = list()
    for i in range(NVIEWS):
        dic = {
            "id_view" : i,
            "codes" : codes,
            "codes_test" : codes_test,
            "model" : models[i],
            "links" : links,
            "train_dataset" : train_datasets[i],
            "test_dataset" : test_datasets[i],
            "options" : options
        }
        args.append(dic)
    return args

# =====================================================================

def getWeightedOutput(i, codes, links, weights):
    w_partial_reconstr = list()
    for j in range(len(codes)):
        if i != j :
            code_externe = codes[j]
            partial_reconstr = links[j][i](code_externe)*weights[j]
            w_partial_reconstr.append(partial_reconstr)

    mean_reconstr = ft.reduce(lambda x, y: x+y, w_partial_reconstr)

    return mean_reconstr

# =====================================================================

def get_args_to_map_links3(codes, codes_test, train_datasets, test_datasets, options):
    NVIEWS = len(codes)
    args = list()
    for i in range(NVIEWS):
        for j in range(NVIEWS):
            dic = {
                "id_in" : i,
                "id_out" : j,
                "data_in" : train_datasets[i],
                "data_out" : codes[j],
                "test_in" : test_datasets[i],
                "test_out" : codes_test[j],
                "options" : options,
            }
            args.append(dic)
    return args

# =====================================================================

def get_args_to_map_links4(codes, codes_test, train_datasets, test_datasets, options):
    NVIEWS = len(codes)
    args = list()
    for i in range(NVIEWS):
        for j in range(NVIEWS):
            dic = {
                "id_in" : i,
                "id_out" : j,
                "data_in" : codes[i],
                "data_out" : train_datasets[j],
                "test_in" : codes_test[i],
                "test_out" : test_datasets[j],
                "options" : options,
            }
            args.append(dic)
    return args

# =====================================================================

def get_args_to_map_weights3(train_datasets, test_datasets, models, links, options):
    NVIEWS = len(train_datasets)
    args = list()
    for i in range(NVIEWS):
        dic = {
            "id_view" : i,
            "model" : models[i],
            "links" : links,
            "train_datasets" : train_datasets,
            "test_datasets" : test_datasets,
            "options" : options
        }
        args.append(dic)
    return args

# =====================================================================

def get_args_to_map_weights4(codes, codes_test, train_datasets, test_datasets, models, links, options):
    NVIEWS = len(train_datasets)
    args = list()
    for i in range(NVIEWS):
        dic = {
            "id_view" : i,
            "links" : links,
            "codes" : codes,
            "codes_test" : codes_test,
            "train_dataset" : train_datasets[i],
            "test_dataset" : test_datasets[i],
            "options" : options
        }
        args.append(dic)
    return args

# =====================================================================

def getWeightedInputCodes3(i, datasets, links, weights):
    w_codes = list()
    for j in range(len(datasets)):
        if i != j :
            data_externe = datasets[j]
            code_interne = links[j][i](data_externe)*weights[j,:]
            w_codes.append(code_interne)

    code_moyen = ft.reduce(lambda x, y: x+y, w_codes)
    return code_moyen

# =====================================================================

def get_weighted_outputs(i, codes, links, weights):
    w_output = list()
    for j in range(len(codes)):
        if i != j :
            code_externe = codes[j]
            data_interne = links[j][i](code_externe)
            data_interne *= weights[j,:]
            w_output.append(data_interne)

    data_moyen = ft.reduce(lambda x, y: x+y, w_output)
    return data_moyen

# =====================================================================

def update_weights(i, codes, target, links, weights):
    new_weights = Variable(weights.data)
    for j in range(len(codes)):
        if j != i:
            code_externe = codes[j]
            data_interne = links[j][i](code_externe)
            w_output = list()
            for j2 in range(len(codes)):
                if j2 != j and j2 != i:
                    code_externe2 = codes[j2]
                    data_interne2 = links[j2][i](code_externe2)
                    data_interne2 *= weights[j2,:]
                    w_output.append(data_interne2)
            data_moyen = ft.reduce(lambda x, y: x+y, w_output)
            delta = target - data_moyen
            numerateur = data_interne * delta
            numerateur = torch.sum(numerateur, dim=0)
            denominateur = data_interne * data_interne
            denominateur = torch.sum(denominateur, dim=0)
            new_weights[j,:] = numerateur/denominateur
    return new_weights

# =====================================================================

def get_args_to_map_classifiers(train_datasets, test_datasets, train_labels, test_labels, options):
    NVIEWS = len(train_datasets)
    args = list()
    for i in range(NVIEWS):
        dic = {
            "id_in" : i,
            "data_in" : train_datasets[i],
            "data_out" : train_labels,
            "test_in" : test_datasets[i],
            "test_out" : test_labels,
            "options" : options
        }
        args.append(dic)
    return args

# =====================================================================

def get_dim_sparse_file(filename) :
    dimData = 0
    for line in open(filename) :
        if line[0] != "#" and len(line) > 1 :
            line = line.split(" ", 1)
        # In case an instance with all zero features
            if len(line) == 1: 
                continue
            label, features = line
            features = features.split(" ")
            for index_feature, e in enumerate(features):
                ind, val = e.split(":")
                dimData = max(dimData, int(ind))
    print("dimData : " + str(dimData))
    return dimData

# =====================================================================

def labels_as_matrix(labels):
    n_categories = pd.get_dummies(labels).values.shape[1]
    n = pd.Series(labels, dtype="category")
    n = n.cat.rename_categories(range(n_categories))
    n = asarray(n)
    labels = set(labels)
    return((n, labels))

# =====================================================================

def crit_per_feature(data, target):
    diff = torch.abs(data.data - target.data)
    means = torch.mean(diff, dim = 0)
    return means

# =====================================================================

def new_loss(data, target):
    diff = (data - target)/target
    relative = torch.mean(torch.abs(diff))
    return relative

# =====================================================================

def normalize_weights(weights, id_view):
    weights.requires_grad = False
    for i in range(weights.shape[1]):
        weights[:,i] = weights[:,i] / (sum(weights[:,i])-weights[id_view,i])
    weights = Variable(weights.data, requires_grad = True)
    return weights

# =====================================================================

def extract_results(l):
    objects = []
    results = []
    for t in l :
        objects.append(t[0])
        results.append(t[1])
    return objects, results

# =====================================================================

def get_iterator(filename, chunksize, datatype):
    data = pd.read_csv(filename, chunksize=chunksize)
    if datatype == "float" :
        data = map(lambda chunk: Variable(torch.from_numpy(chunk.values).float(), requires_grad=False), data)
    elif datatype == "long" :
        data = map(lambda chunk: Variable(torch.from_numpy(chunk.values).long().squeeze(), requires_grad=False), data)
    else :
        sys.exit("Error, type {} undefined".format(datatype)) 
    return data
  














