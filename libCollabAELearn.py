import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from copy import deepcopy
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
        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=options["LEARNING_RATE_AE"], momentum=options["MOMENTUM"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        # LEARNING
        test_fail = 0
        min_test = float("inf")
        min_model = object()
        for epoch in range(options["NSTEPS"]):

            # Test information
            outputs = net(dataset_test)
            loss = criterion(outputs, dataset_test)
            if epoch % options["VERBOSE_STEP"] == 0 and options["VERBOSE"] and id_net == 0: 
                print("Test Loss " + str(epoch) + " : " + str(loss.data[0]))

            # Stop when test error is increasing
            if loss.data[0] > min_test :
                test_fail += 1
                if test_fail > options["PATIENCE"] :
                    if options["VERBOSE"] and id_net == 0: print("Stop : test error increasing")
                    net = deepcopy(min_model)
                    break
            else :
                min_test = loss.data[0]
                test_fail = 0
                min_model = deepcopy(net)

            optimizer.zero_grad()
            
            # Train information
            outputs = net(dataset)
            loss = criterion(outputs, dataset)

            # Parameters optimization
            loss.backward()
            optimizer.step()
            scheduler.step(loss.data[0])

        outputs = net(dataset_test)
        loss = criterion(outputs, dataset_test)
        print("\tView " + str(id_net) + " - test loss (MSE) : " + str(loss.data[0]))

        return net

# =====================================================================

def learn_LinkNet(args):
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
        net = LinkNet( [dimData_in] + options["LAYERS_LINKS"] + [dimData_out] )
        criterion = nn.MSELoss()
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
            outputs = net(data_test_in)
            loss = criterion(outputs, data_test_out)
            if epoch % options["VERBOSE_STEP"] == 0 and options["VERBOSE"] and i == 0 and j == 1: 
                print("Test Loss " + str(epoch) + " : " + str(loss.data[0]))

            # Stop if test error is increasing
            if loss.data[0] > min_test :
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
            outputs = net(data_in)
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
    w = torch.FloatTensor(NVIEWS).zero_()+1/(NVIEWS-1)
    weights = (Variable(w, requires_grad=True))
    criterion = nn.MSELoss()
    if options["LEARN_WEIGHTS"] :

        optimizer = optim.SGD([weights], lr=options["LEARNING_RATE_WEIGHTS"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)
        
        for epoch in range(options["NSTEPS_WEIGHTS"]):
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

    print("\tReconstruction view " + str(id_view) + " - test loss (MSE) : " + str(loss.data[0]))
    print("\tWeights view " + str(id_view) + " : " + str(weights[:].data.numpy()))
    return(weights)

# =====================================================================

def learn_ClassifierNet(args):
    i = args["id_in"]
    options = args["options"]

    data_in = args["data_in"]
    data_out = args["data_out"]
    data_test_in = args["test_in"]
    data_test_out = args["test_out"]
    dimData_in = data_in.size()[1]
    dimData_out = torch.max(data_out) + 1

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

        # Test information
        outputs = net(data_test_in)
        loss = criterion(outputs, data_test_out)
        if epoch % options["VERBOSE_STEP"] == 0 and options["VERBOSE"] and i == 0 : 
            print("Test Loss " + str(epoch) + " : " + str(loss.data[0]))

        # Stop if test error is increasing
        if loss.data[0] > min_test :
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
        outputs = net(data_in)
        loss = criterion(outputs, data_out)

        # Parameters optimization
        loss.backward()
        optimizer.step()
        scheduler.step(loss.data[0])

    # outputs = net(data_test_in)
    # loss = criterion(outputs, data_test_out)
    # print("\tClassifier " + str(i) + " - test loss : " + str(loss.data[0]))

    predictions = net.getClasses(data_test_in)
    accuracy = torch.sum(torch.eq(predictions, data_test_out)).float()/len(data_test_out)
    accuracy = str(accuracy.data[0])

    print("\tClassifier " + str(i) + " - accuracy : " + accuracy)

    return net

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

    # LEARNING OF THE LINKS
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
