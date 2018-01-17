from libCollabAE import *
from multiprocessing import Pool
from collections import Counter

def learnCollabSystem(train_datasets, test_datasets, options) :

	p = Pool(4)

	# PARAMETERS
	NVIEWS = len(train_datasets)

	# LEARNING ALL THE MODELS AND GET THE CODES
	args = get_args_to_map_AE(train_datasets, test_datasets, options)
	models = p.map(learn_AENet, args)

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
	args = get_args_to_map_links(codes, codes_test, options)
	links_tmp = p.map(learn_LinkNet, args)

	links = list()
	for i in range(NVIEWS):
		links.append(list())
		for j in range(NVIEWS):
			links[i].append(links_tmp[i*NVIEWS+j])

	print("\n")

	args = get_args_to_map_weights(train_datasets, test_datasets, models, links, codes, codes_test, options)
	weights = p.map(learn_weights_code, args)

	print("Done")

# =====================================================================

def learnCollabSystem2(train_datasets, test_datasets, options) :

	p = Pool(4)

	# PARAMETERS
	NVIEWS = len(train_datasets)

	# LEARNING ALL THE MODELS AND GET THE CODES
	args = get_args_to_map_AE(train_datasets, test_datasets, options)
	models = p.map(learn_AENet, args)

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

	args = get_args_to_map_links2(codes, codes_test, train_datasets, test_datasets, options)
	links_tmp = p.map(learn_LinkNet, args)

	links = list()
	for i in range(NVIEWS):
		links.append(list())
		for j in range(NVIEWS):
			links[i].append(links_tmp[i*NVIEWS+j])

	print("\n")

	args = get_args_to_map_weights(train_datasets, test_datasets, models, links, codes, codes_test, options)
	weights = p.map(learn_weights_code2, args)

	print("Done")

# =====================================================================

def learnCollabSystem3(train_datasets, test_datasets, options) :

	p = Pool(4)

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
		classifiers = p.map(learn_ClassifierNet, args)

	# LEARNING ALL THE MODELS AND GET THE CODES
	print("Learning autoencoders...")
	args = get_args_to_map_AE(train_datasets, test_datasets, options)
	models = p.map(learn_AENet, args)

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
	links_tmp = p.map(learn_LinkNet, args)

	links = list()
	for i in range(NVIEWS):
		links.append(list())
		for j in range(NVIEWS):
			links[i].append(links_tmp[i*NVIEWS+j])

	print("Learning weights...")
	args = get_args_to_map_weights3(train_datasets, test_datasets, models, links, options)
	weights = p.map(learn_weights_code3, args)

	cs = CollabSystem(models, links, weights)

	print("Classification of the reconstruction")
	for i in range(NVIEWS):
		output = cs.forward(i, test_datasets)
		output = classifiers[i].getClasses(output)
		output = torch.sum(torch.eq(output, test_labels)).float()/len(test_labels)
		print("\tAccuracy view " + str(i) + " : " + str(output.data[0]))

	print("Done")