from libCollabAE import *
from multiprocessing import Pool

def learnCollabSystem(train_datasets, test_datasets, options) :

	inf = float("inf")
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
	sys.exit()

	links = list()
	for i in range(NVIEWS):
		links.append(list())
		for j in range(NVIEWS):
			links[i].append(links_tmp[i*NVIEWS+j])

	print("\n")

	# TESTING THE RECONSTRUCTION
	# PROTO WEIGTHING WITH GRAD
	w = torch.FloatTensor(NVIEWS,NVIEWS).zero_()+1/(NVIEWS-1)
	weights = (Variable(w, requires_grad=True))
	criterion = nn.MSELoss()

	for i in range(NVIEWS):
		print("Reconstruction view " + str(i))
		optimizer = optim.SGD([weights], lr=LEARNING_RATE_WEIGHTS)
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
		
		for epoch in range(NSTEPS_WEIGHTS):
			optimizer.zero_grad()

			code_moyen = getWeightedInputCodes(i, models, links, train_datasets, weights)
			indiv_reconstruit = models[i].decode(code_moyen)
			
			loss = criterion(indiv_reconstruit, train_datasets[i])
			loss.backward()
			optimizer.step()
			scheduler.step(loss.data[0])

			if epoch % VERBOSE_STEP == 0 and VERBOSE:
				code_test_moyen = getWeightedInputCodes(i, models, links, test_datasets, weights)
				indiv_reconstruit = models[i].decode(code_test_moyen)
				loss = criterion(indiv_reconstruit, test_datasets[i])
				print("Reconst. Test loss " + str(epoch) + " : " + str(loss.data[0]))

		code_test_moyen = getWeightedInputCodes(i, models, links, test_datasets, weights)
		indiv_reconstruit = models[i].decode(code_test_moyen)
		loss = criterion(indiv_reconstruit, test_datasets[i])
		print("\ttest loss : " + str(loss.data[0]))
		print("\n")

	print("Weights")
	print(weights[:,:])