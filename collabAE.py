from libCollabAE import *
from multiprocessing import Pool

def learnCollabSystem(train_datasets, test_datasets, options) :

	p = Pool(4)

	# PARAMETERS
	VERBOSE = options["VERBOSE"]
	VERBOSE_STEP = options["VERBOSE_STEP"]
	NVIEWS = len(train_datasets)

	# HYPERPARAMETERS
	NSTEPS = options["NSTEPS"]
	NSTEPS_WEIGHTS = options["NSTEPS_WEIGHTS"]
	LAYERS_AE = options["LAYERS_AE"]
	LAYERS_LINKS = options["LAYERS_LINKS"]
	LEARNING_RATE_AE = options["LEARNING_RATE_AE"]
	LEARNING_RATE_LINKS = options["LEARNING_RATE_LINKS"]
	LEARNING_RATE_WEIGHTS = options["LEARNING_RATE_WEIGHTS"]
	MOMENTUM = options["MOMENTUM"]
	PATIENCE = options["PATIENCE"]

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
	VERBOSE = options["VERBOSE"]
	VERBOSE_STEP = options["VERBOSE_STEP"]
	NVIEWS = len(train_datasets)

	# HYPERPARAMETERS
	NSTEPS = options["NSTEPS"]
	NSTEPS_WEIGHTS = options["NSTEPS_WEIGHTS"]
	LAYERS_AE = options["LAYERS_AE"]
	LAYERS_LINKS = options["LAYERS_LINKS"]
	LEARNING_RATE_AE = options["LEARNING_RATE_AE"]
	LEARNING_RATE_LINKS = options["LEARNING_RATE_LINKS"]
	LEARNING_RATE_WEIGHTS = options["LEARNING_RATE_WEIGHTS"]
	MOMENTUM = options["MOMENTUM"]
	PATIENCE = options["PATIENCE"]

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
	VERBOSE = options["VERBOSE"]
	VERBOSE_STEP = options["VERBOSE_STEP"]
	NVIEWS = len(train_datasets)

	# HYPERPARAMETERS
	NSTEPS = options["NSTEPS"]
	NSTEPS_WEIGHTS = options["NSTEPS_WEIGHTS"]
	LAYERS_AE = options["LAYERS_AE"]
	LAYERS_LINKS = options["LAYERS_LINKS"]
	LEARNING_RATE_AE = options["LEARNING_RATE_AE"]
	LEARNING_RATE_LINKS = options["LEARNING_RATE_LINKS"]
	LEARNING_RATE_WEIGHTS = options["LEARNING_RATE_WEIGHTS"]
	MOMENTUM = options["MOMENTUM"]
	PATIENCE = options["PATIENCE"]

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
	args = get_args_to_map_links3(codes, codes_test, train_datasets, test_datasets, options)
	links_tmp = p.map(learn_LinkNet, args)

	links = list()
	for i in range(NVIEWS):
		links.append(list())
		for j in range(NVIEWS):
			links[i].append(links_tmp[i*NVIEWS+j])

	print("\n")

	args = get_args_to_map_weights3(train_datasets, test_datasets, models, links, options)
	weights = p.map(learn_weights_code3, args)

	print("Done")