import sys
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.cluster as cluster

from copy import deepcopy
from itertools import tee
from collections import Counter, Iterator
from torch.autograd import Variable

from libCollabAEClasses import *
from libCollabAEUtils import *


def learn_AENet(args):
        dataset = args["train"]
        dataset_test = args["test"]
        options = args["options"]
        id_net = args["id_net"]

        dimData = dataset.size()[1]

        # MODEL DEFINITION
        net = AENet( [dimData] + options["LAYERS_AE"] )
        criterion = options["LOSS_METHOD"]
        optimizer = optim.SGD(net.parameters(), lr=options["LEARNING_RATE_AE"], momentum=options["MOMENTUM"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        # LEARNING
        test_fail = 0
        min_test = float("inf")
        min_model = object()


        for epoch in range(options["NSTEPS"]):
            # Modification de dataset_test pour gérer les itérables
            if isinstance(dataset, Iterator):
                copy_dataset = tee(dataset, n=1)
            else :
                copy_dataset = iter([dataset])
            # Test information
            outputs = net(dataset_test)
            loss = criterion(outputs, dataset_test)
            if epoch % options["VERBOSE_STEP"] == 0 and options["VERBOSE"] and id_net == 0: 
                print("Test Loss " + str(epoch) + " : " + str(loss.data[0]))

            # Stop when test error is increasing
            if loss.data[0] >= min_test :
                test_fail += 1
                if test_fail > options["PATIENCE"] :
                    if options["VERBOSE"] and id_net == 0: print("Stop : test error increasing")
                    net = deepcopy(min_model)
                    break
            else :
                min_test = loss.data[0]
                test_fail = 0
                min_model = deepcopy(net)

            for chunk in copy_dataset :
                optimizer.zero_grad()
                
                # Train information
                outputs = net(chunk)
                loss = criterion(outputs, chunk)

                # Parameters optimization
                loss.backward()
                optimizer.step()
                scheduler.step(loss.data[0])

        outputs = net(dataset_test)
        loss = criterion(outputs, dataset_test)
        print("\tView " + str(id_net) + " - test loss (MSE) : " + str(loss.data[0]))

        return net, loss.data[0]

# =====================================================================

def learn_LinkNet(args):
    i = args["id_in"]
    j = args["id_out"]
    options = args["options"]

    if i == j :
        return [], 0
    else :
        data_in = args["data_in"]
        data_out = args["data_out"]
        data_test_in = args["test_in"]
        data_test_out = args["test_out"]


        # Modification de dataset_test pour récupérer les dimensions
        if isinstance(data_in, Iterator):
            copy_data_in = tee(data_in, n=1)
            copy_data_out = tee(data_out, n=1)
        else :
            copy_data_in = iter([data_in])
            copy_data_out = iter([data_out])
        dimData_in = copy_data_in.__next__().size()[1]
        dimData_out = copy_data_out.__next__().size()[1]

        # DEFINE THE MODEL
        net = LinkNet( [dimData_in] + options["LAYERS_LINKS"] + [dimData_out], options["clampOutput"] )
        criterion = options["LOSS_METHOD"]
        optimizer = optim.SGD( net.parameters(), \
            lr=options["LEARNING_RATE_LINKS"], \
            momentum=options["MOMENTUM"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        # LEARNING
        test_fail = 0
        min_test = float("inf")
        min_model = object()
        for epoch in range(options["NSTEPS"]):
            # Modification de dataset_test pour gérer les itérables
            if isinstance(data_in, Iterator):
                copy_data_in = tee(data_in, n=1)
                copy_data_out = tee(data_out, n=1)
            else :
                copy_data_in = iter([data_in])
                copy_data_out = iter([data_out])

            # Test information
            outputs = net(data_test_in)
            loss = criterion(outputs, data_test_out)
            if epoch % options["VERBOSE_STEP"] == 0 and options["VERBOSE"] and i == 0 and j == 1: 
                print("Test Loss " + str(epoch) + " : " + str(loss.data[0]))

            # Stop if test error is increasing
            if loss.data[0] >= min_test :
                test_fail += 1
                if test_fail > options["PATIENCE"] :
                    if options["VERBOSE"] and i == 0 : print("Stop : test error increasing")
                    net = deepcopy(min_model)
                    break
            else :
                min_test = loss.data[0]
                test_fail = 0
                min_model = deepcopy(net)

            for chunk_in, chunk_out in zip(copy_data_in, copy_data_out) :
                optimizer.zero_grad()
                
                # Train information
                outputs = net(chunk_in)
                loss = criterion(outputs, chunk_out)

                # Parameters optimization
                loss.backward()
                optimizer.step()
                scheduler.step(loss.data[0])

        outputs = net(data_test_in)
        loss = criterion(outputs, data_test_out)
        print("\tLink " + str(i) + " ~ " + str(j) + " - test loss (MSE) : " + str(loss.data[0]))

        return net, loss.data[0]

# =====================================================================

def learn_GeminiLinkNet(args):
    i = args["id_in"]
    j = args["id_out"]
    options = args["options"]

    if i == j :
        return []
    else :
        data_in = args["data_in"]
        data_out = args["data_out"]
        data_test_in = args["test_in"]
        data_test_out = args["test_out"]

        dimData_in = data_in.size()[1]
        dimData_out = data_out.size()[1]

        # DEFINE THE MODEL
        net = GeminiLinkNet( [[dimData_in, dimData_out]] + options["LAYERS_LINKS"] + [dimData_out], options["clampOutput"] )
        criterion = options["LOSS_METHOD"]
        optimizer = optim.SGD( net.parameters(), \
            lr=options["LEARNING_RATE_LINKS"], \
            momentum=options["MOMENTUM"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        # LEARNING
        test_fail = 0
        min_test = float("inf")
        min_model = object()
        for epoch in range(options["NSTEPS"]):

            # Test information
            outputs = net(x1 = data_test_in, x2 = data_test_out)
            loss = criterion(outputs, data_test_out)
            outputs_half = net(x1 = data_test_in)
            loss2 = criterion(outputs_half, data_test_out)

            if epoch % options["VERBOSE_STEP"] == 0 and options["VERBOSE"] and i == 0 and j == 1: 
                print("Test Loss " + str(epoch) + " : " + str(loss.data[0]))
                print("Test Loss (half) " + str(epoch) + " : " + str(loss2.data[0]))

            # Stop if test error is increasing
            if loss.data[0] >= min_test :
                test_fail += 1
                if test_fail > options["PATIENCE"] :
                    if options["VERBOSE"] and i == 0 : print("Stop : test error increasing")
                    net = deepcopy(min_model)
                    break
            else :
                min_test = loss.data[0]
                test_fail = 0
                min_model = deepcopy(net)

            optimizer.zero_grad()
            
            # Train information
            outputs = net(x1 = data_in, x2 = data_out)
            loss = criterion(outputs, data_out)

            # Parameters optimization
            loss.backward()
            optimizer.step()
            scheduler.step(loss.data[0])

        outputs = net(data_test_in)
        loss = criterion(outputs, data_test_out)
        print("\tLink " + str(i) + " ~ " + str(j) + " - test loss (MSE) : " + str(loss.data[0]))

        return net

# =====================================================================

def learn_weights_code3(args):
    id_view = args["id_view"]

    model = args["model"]
    links = args["links"]

    train_datasets = args["train_datasets"]
    test_datasets = args["test_datasets"]

    options = args["options"]

    NVIEWS = len(train_datasets)

    # TESTING THE RECONSTRUCTION
    # PROTO WEIGTHING WITH GRAD
    dimCode = links[0][1].forward(test_datasets[0]).shape[1]
    w = torch.FloatTensor(NVIEWS, dimCode).zero_()+1/(NVIEWS-1)
    weights = (Variable(w, requires_grad=True))
    criterion = options["LOSS_METHOD"]

    # Variables for early stop
    best_error = float("inf")
    test_fail = 0

    if options["LEARN_WEIGHTS"] :

        optimizer = optim.SGD([weights], lr=options["LEARNING_RATE_WEIGHTS"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300)
        
        for epoch in range(options["NSTEPS_WEIGHTS"]):
            # Test data to test early stop
            code_moyen = getWeightedInputCodes3(id_view, test_datasets, links, weights)
            indiv_reconstruit = model.decode(code_moyen)
            loss = criterion(indiv_reconstruit, test_datasets[id_view])
            if loss.data[0] > best_error :
                test_fail += 1
                if test_fail >= options["PATIENCE"] :
                    if options["VERBOSE"] and id_view == 0 : print("Stop : test error increasing")
                    weights = min_weights
                    break
            else :
                best_error = loss.data[0]
                test_fail = 0
                min_weights = (Variable(weights.data, requires_grad=True))         

            # Train data to learn weights
            optimizer.zero_grad()
            code_moyen = getWeightedInputCodes3(id_view, train_datasets, links, weights)
            indiv_reconstruit = model.decode(code_moyen)
            
            loss = criterion(indiv_reconstruit, train_datasets[id_view])
            loss.backward()
            optimizer.step()
            scheduler.step(loss.data[0])

            if epoch % options["VERBOSE_STEP"] == 0 and options["VERBOSE"] and id_view == 0:
                code_test_moyen = getWeightedInputCodes3(id_view, test_datasets, links, weights)
                indiv_reconstruit = model.decode(code_test_moyen)
                loss = criterion(indiv_reconstruit, test_datasets[id_view])
                print("Reconst. Test loss " + str(epoch) + " : " + str(loss.data[0]))

    code_test_moyen = getWeightedInputCodes3(id_view, test_datasets, links, weights)
    indiv_reconstruit = model.decode(code_test_moyen)
    loss = criterion(indiv_reconstruit, test_datasets[id_view])
    loss2 = crit_per_feature(indiv_reconstruit, test_datasets[id_view])
    mean_per_feat = torch.mean(torch.abs(test_datasets[id_view]), dim=0).data
    delta = torch.abs((loss2 - mean_per_feat)/mean_per_feat)

    print("\tReconstruction view " + str(id_view) + " - test loss (MSE) : " + str(loss.data[0]))
    print("\t\tMean relative error per feature : " + str(delta))
    print("\t\tWeights view : " + str(weights[:,:].data))
    return(weights)

# =====================================================================

def learn_weights_code4(args):
    id_view = args["id_view"]

    links = args["links"]

    codes = args["codes"]
    codes_test = args["codes_test"]
    train_dataset = args["train_dataset"]
    test_dataset = args["test_dataset"]

    options = args["options"]

    NVIEWS = len(codes)

    print(codes)

    # TESTING THE RECONSTRUCTION
    # PROTO WEIGTHING WITH GRAD
    w = torch.FloatTensor(NVIEWS, test_dataset.shape[1]).zero_()+1/(NVIEWS-1)
    weights = (Variable(w, requires_grad=True))
    criterion = options["LOSS_METHOD"]

    # Variables for early stop
    best_error = float("inf")
    test_fail = 0

    if options["LEARN_WEIGHTS"] :

        optimizer = optim.SGD([weights], lr=options["LEARNING_RATE_WEIGHTS"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)
        
        for epoch in range(options["NSTEPS_WEIGHTS"]):
            if isinstance(codes[0], Iterator):
                codes_tmp = [tee(code, 1) for code in codes]
                train_dataset_tmp = tee(train_dataset, 1)
            else :
                codes_tmp = [iter([code]) for code in codes]
                train_dataset_tmp = iter([train_dataset])

            indiv_reconstruit = get_weighted_outputs(id_view, codes_test, links, weights)
            loss = criterion(indiv_reconstruit, test_dataset)

            if loss.data[0] > best_error :
                test_fail += 1
                if test_fail >= options["PATIENCE"] :
                    if options["VERBOSE"] and id_view == 0 : print("Stop : test error increasing")
                    weights = min_weights
                    break
            else :
                best_error = loss.data[0]
                test_fail = 0
                min_weights = (Variable(weights.data, requires_grad=True))  

            if epoch % options["VERBOSE_STEP"] == 0 and options["VERBOSE"] and id_view == 0:
                print("Reconst. Test loss " + str(epoch) + " : " + str(loss.data[0]))

            for chunk_codes in zip(*codes_tmp, train_dataset_tmp):
                optimizer.zero_grad()

                indiv_reconstruit = get_weighted_outputs(id_view, chunk_codes[:-1], links, weights)
                loss = criterion(indiv_reconstruit, chunk_codes[-1])
                
                loss.backward()
                optimizer.step()
                scheduler.step(loss.data[0])


    # weights = normalize_weights(weights, id_view)
    indiv_reconstruit = get_weighted_outputs(id_view, codes_test, links, weights)
    loss = criterion(indiv_reconstruit, test_dataset)
    # loss2 = crit_per_feature(indiv_reconstruit, test_dataset)
    # mean_per_feat = torch.mean(torch.abs(test_dataset), dim=0).data
    # delta = torch.abs((loss2 - mean_per_feat)/mean_per_feat)
    l = torch.abs((test_dataset - indiv_reconstruit)/test_dataset)
    # print(l)

    l = torch.median(l)
    print("\tReconstruction view " + str(id_view) + " - test loss (MSE) : " + str(loss.data[0]))
    print("\tWeights view : " + str(weights[:,:]))
    # print("\tMean Relative Error : " + str(l.data.numpy()[0]))
    return weights, (loss.data[0], l.data[0])

# =====================================================================

def learn_ClassifierNet(args):
    i = args["id_in"]
    options = args["options"]

    data_in = args["data_in"]
    data_out = args["data_out"]
    data_test_in = args["test_in"]
    data_test_out = args["test_out"]

    # Modification de dataset_test pour récupérer les dimensions
    if isinstance(data_in, Iterator):
        copy_data_in = tee(data_in, n=1)
        copy_data_out = tee(data_out, n=1)
    else :
        copy_data_in = iter([data_in])
        copy_data_out = iter([data_out])
    dimData_in = copy_data_in.__next__().size()[1]
    dimData_out = torch.max(copy_data_out.__next__()) + 1


    # DEFINE THE MODEL
    net = ClassifNet( [dimData_in] + options["LAYERS_CLASSIF"] + [dimData_out.data[0]] )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD( net.parameters(), \
        lr=options["LEARNING_RATE_CLASSIF"], \
        momentum=options["MOMENTUM"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30)

    # LEARNING
    test_fail = 0
    min_test = float("inf")
    min_model = object()
    for epoch in range(options["NSTEPS"]):
        # Modification de dataset_test pour gérer les itérables
        if isinstance(data_in, Iterator):
            copy_data_in = tee(data_in, n=1)
            copy_data_out = tee(data_out, n=1)
        else :
            copy_data_in = iter([data_in])
            copy_data_out = iter([data_out])

        # Test information
        outputs = net(data_test_in)
        loss = criterion(outputs, data_test_out)
        if epoch % options["VERBOSE_STEP"] == 0 and options["VERBOSE"] and i == 0 : 
            print("Test Loss " + str(epoch) + " : " + str(loss.data[0]))

        # Stop if test error is increasing
        if loss.data[0] >= min_test :
            test_fail += 1
            if test_fail > options["PATIENCE"] :
                if options["VERBOSE"] and i == 0 : print("Stop : test error increasing")
                net = deepcopy(min_model)
                break
        else :
            min_test = loss.data[0]
            test_fail = 0
            min_model = deepcopy(net)

        for chunk_in, chunk_out in zip(copy_data_in, copy_data_out) :
            optimizer.zero_grad()
            
            # Train information
            outputs = net(chunk_in)
            loss = criterion(outputs, chunk_out)

            # Parameters optimization
            loss.backward()
            optimizer.step()
            scheduler.step(loss.data[0])

    # outputs = net(data_test_in)
    # loss = criterion(outputs, data_test_out)
    # print("\tClassifier " + str(i) + " - test loss : " + str(loss.data[0]))

    predictions = net.getClasses(data_test_in)
    accuracy = torch.sum(torch.eq(predictions, data_test_out)).float()/len(data_test_out)

    print("\tClassifier " + str(i) + " - accuracy : " + str(accuracy.data[0]))

    return net, accuracy.data[0]

# =====================================================================

def learn_Clustering(args):
    model = cluster.SpectralClustering(n_clusters = 3)
    model.fit(args["data_in"].data.numpy())
    print("nClusters view " + str(args["id_in"]) + " : " + str(len(set(model.labels_))))
    return model

# =====================================================================

def learnCollabSystem3(train_datasets, test_datasets, options) :

    # p = Pool(4)

    # PARAMETERS
    NVIEWS = len(train_datasets)

    if "train_labels" in options :
        print("Learning classifiers...")
        train_labels = options["train_labels"]
        tmp = Counter(train_labels.data.numpy())
        train_apriori = float(tmp.most_common(1)[0][1])/len(train_labels.data)

        test_labels = options["test_labels"]
        tmp = Counter(test_labels.data.numpy())
        test_apriori = float(tmp.most_common(1)[0][1])/len(test_labels.data)
        
        print("\tTrain a priori : " + str(train_apriori))
        print("\tTest a priori : " + str(test_apriori))

        args = get_args_to_map_classifiers(train_datasets, test_datasets, train_labels, test_labels, options)
        # classifiers = p.map(learn_ClassifierNet, args)
        classifiers = list(learn_ClassifierNet(arg) for arg in args)

    # LEARNING ALL THE MODELS AND GET THE CODES
    print("Learning autoencoders...")
    args = get_args_to_map_AE(train_datasets, test_datasets, options)
    # models = p.map(learn_AENet, args)
    models = list(learn_AENet(arg) for arg in args)

    codes = list()
    codes_test = list()
    for i in range(NVIEWS):
        # Codes gathering
        code = models[i].encode(train_datasets[i])
        code = Variable(code.data, requires_grad = False)
        codes.append(code)

        code_test = models[i].encode(test_datasets[i])
        code_test = Variable(code_test.data, requires_grad = False)
        codes_test.append(code_test)

    # LEARNING OF THE LINKS
    print("Learnings links...")
    args = get_args_to_map_links3(codes, codes_test, train_datasets, test_datasets, options)
    # links_tmp = p.map(learn_LinkNet, args)
    links_tmp = list(learn_LinkNet(arg) for arg in args)

    links = list()
    for i in range(NVIEWS):
        links.append(list())
        for j in range(NVIEWS):
            links[i].append(links_tmp[i*NVIEWS+j])

    print("Learning weights and final reconstruction...")
    args = get_args_to_map_weights3(train_datasets, test_datasets, models, links, options)
    # weights = p.map(learn_weights_code3, args)
    weights = list(learn_weights_code3(arg) for arg in args)

    cs = CollabSystem(models, links, weights)

    print("Classification of the reconstruction")
    for i in range(NVIEWS):
        output = cs.forward(i, test_datasets)
        output = classifiers[i].getClasses(output)
        output = torch.sum(torch.eq(output, test_labels)).float()/len(test_labels)
        print("\tAccuracy view " + str(i) + " : " + str(output.data[0]))

# =====================================================================

def learnCollabSystem4(train_datasets, test_datasets, options) :

    results = {}

    # PARAMETERS
    NVIEWS = len(train_datasets)

    if "train_labels" in options :
        print("Learning classifiers...")
        train_labels = options["train_labels"]
        tmp = Counter(train_labels.data.numpy())
        train_apriori = float(tmp.most_common(1)[0][1])/len(train_labels.data)

        test_labels = options["test_labels"]
        tmp = Counter(test_labels.data.numpy())
        test_apriori = float(tmp.most_common(1)[0][1])/len(test_labels.data)
        
        print("\tTrain a priori : " + str(train_apriori))
        print("\tTest a priori : " + str(test_apriori))
        results["error_apriori_train"] = train_apriori
        results["error_apriori_test"] = test_apriori

        args = get_args_to_map_classifiers(train_datasets, test_datasets, train_labels, test_labels, options)
        # clusterings = list(learn_Clustering(arg) for arg in args)
        classifiers = list(learn_ClassifierNet(arg) for arg in args)
        classifiers, results["error_classifiers_test"] = extract_results(classifiers)

    # LEARNING ALL THE MODELS AND GET THE CODES
    print("Learning autoencoders...")
    args = get_args_to_map_AE(train_datasets, test_datasets, options)
    models = list(learn_AENet(arg) for arg in args)
    models, results["error_autoencoders_test"] = extract_results(models)

    codes = list()
    codes_test = list()
    for i in range(NVIEWS):
        train = train_datasets[i]
        test = test_datasets[i]

            # Codes gathering
        if isinstance(train, Iterator):
            code = map(lambda chunk: models[i].encode(chunk), train)
            code = map(lambda chunk : Variable(code.data, requires_grad = False))

            code_test = map(lambda chunk: models[i].encode(chunk), train)
            code_test = map(lambda chunk : Variable(code.data, requires_grad = False))
        else :
            code = models[i].encode(train)
            code = Variable(code.data, requires_grad = False)

            code_test = models[i].encode(test)
            code_test = Variable(code_test.data, requires_grad = False)
        codes.append(code)
        codes_test.append(code_test)

    # LEARNING OF THE LINKS
    print("Learnings links...")
    args = get_args_to_map_links4(codes, codes_test, train_datasets, test_datasets, options)
    if options["version"] == 4 :
        links_tmp = list(learn_LinkNet(arg) for arg in args)
        links_tmp, results["error_links_test"] = extract_results(links_tmp)
    elif options["version"] == 5 :
        links_tmp = list(learn_GeminiLinkNet(arg) for arg in args)

    links = list()
    for i in range(NVIEWS):
        links.append(list())
        for j in range(NVIEWS):
            links[i].append(links_tmp[i*NVIEWS+j])

    print("Learning weights and final reconstruction...")
    args = get_args_to_map_weights4(codes, codes_test, train_datasets, test_datasets, models, links, options)
    weights = list(learn_weights_code4(arg) for arg in args)
    weights, results["error_MSE_&_relative_per_feature_test"] = extract_results(weights)
    cs = CollabSystem4(models, links, weights)

    print("Classification of the reconstruction")
    results["error_classifiers_final_test"] = []
    for i in range(NVIEWS):
        output = cs.forward(i, test_datasets)
        output1 = classifiers[i].getClasses(output)
        output1 = torch.sum(torch.eq(output1, test_labels)).float()/len(test_labels)
        results["error_classifiers_final_test"].append(output1.data[0])
        print("\tAccuracy view " + str(i) + " : " + str(output1.data[0]))

    return results

        # output2 = clusterings[i].fit_predict(output.data.numpy())
        # target2 = clusterings[i].fit_predict(test_datasets[i].data.numpy())
        # print("\tClustering view " + str(i) + " : " + str((output2 == target2).sum()/output2.size))
        # print(str(confusion_matrix(output2, target2)))