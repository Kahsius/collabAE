import pickle
import os
import math
import scipy.stats as stats
import numpy as np

def get_results(path) :
    list_results = os.listdir(path)
    K = len(list_results)
    modif = 1.96/math.sqrt(K)
    # print("K : " + str(K))

    data = pickle.load(open(path+list(list_results)[0], "r+b"))
    NVIEWS = len(data["error_classifiers_test"])
    # print("NVIEWS : " + str(NVIEWS))

    classifiers = []
    classifiers_ci = []
    autoencoders = []
    autoencoders_ci = []
    links = []
    links_ci = []
    final_mse = []
    final_mse_ci = []
    final_relative = []
    final_relative_ci = []
    final_classifiers = []
    final_classifiers_ci = []
    for i in range(NVIEWS) :
        classifiers.append([])
        classifiers_ci.append(0)
        autoencoders.append([])
        autoencoders_ci.append(0)
        links.append([])
        links_ci.append([])
        for j in range(NVIEWS) :
            links[i].append([])
            links_ci[i].append(0)
        final_mse.append([])
        final_mse_ci.append(0)
        final_relative.append([])
        final_relative_ci.append(0)
        final_classifiers.append([])
        final_classifiers_ci.append(0)

    for filename in list_results :
        f = open(path+filename, "r+b")
        data = pickle.load(f)
        for i in range(NVIEWS) :
            classifiers[i] += [data["error_classifiers_test"][i]]
            autoencoders[i] += [data["error_autoencoders_test"][i]]
            for j in range(NVIEWS) :
                links[i][j] += [data["error_links_test"][(i-1)*NVIEWS+j]]
            final_mse[i] += [data["error_MSE_&_relative_per_feature_test"][i][0]]
            final_relative[i] += [data["error_MSE_&_relative_per_feature_test"][i][1]]
            final_classifiers[i] += [data["error_classifiers_final_test"][i]]

    def var_to_ci(var, modif) :
        return math.sqrt(var)*modif

    for i in range(NVIEWS) :
        _, _, classifiers[i], classifiers_ci[i], _, _ = stats.describe(classifiers[i])
        classifiers_ci[i] = var_to_ci(classifiers_ci[i], modif)
        _, _, autoencoders[i], autoencoders_ci[i], _, _ = stats.describe(autoencoders[i])
        autoencoders_ci[i] = var_to_ci(autoencoders_ci[i], modif)
        _, _, final_mse[i], final_mse_ci[i], _, _ = stats.describe(final_mse[i])
        final_mse_ci[i] = var_to_ci(final_mse_ci[i], modif)
        _, _, final_relative[i], final_relative_ci[i], _, _ = stats.describe(final_relative[i])
        final_relative_ci[i] = var_to_ci(final_relative_ci[i], modif)
        _, _, final_classifiers[i], final_classifiers_ci[i], _, _ = stats.describe(final_classifiers[i])
        final_classifiers_ci[i] = var_to_ci(final_classifiers_ci[i], modif)
        for j in range(NVIEWS) : 
            _, _, links[i][j], links_ci[i][j], _, _ = stats.describe(links[i][j])
            links_ci[i][j] = var_to_ci(links_ci[i][j], modif)


    # print("Classifiers : " + str(classifiers))
    # print("Classifiers CI : " + str(classifiers_ci))
    # print()
    # print("Autoencoders : " + str(autoencoders))
    # print("Autoencoders CI : " + str(autoencoders_ci))
    # print()
    # print("Links : " + str(links))
    # print("Links CI : " + str(links_ci))
    # print()
    # print("Final MSE : " + str(final_mse))
    # print("Final MSE CI : " + str(final_mse_ci))
    # print()
    # print("Final Relative Error : " + str(final_relative))
    # print("Final Relative Error CI : " + str(final_relative_ci))
    # print()
    # print("Final Classifiers : " + str(final_classifiers))
    # print("Final Classifiers CI : " + str(final_classifiers_ci))

    return classifiers, classifiers_ci, autoencoders, autoencoders_ci, links, links_ci, final_mse, final_mse_ci, final_relative, final_relative_ci, final_classifiers, final_classifiers_ci
