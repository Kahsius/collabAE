import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools as ft
import torch.optim as optim
from sklearn.preprocessing import scale
from copy import deepcopy
from torch.autograd import Variable

# ===================================================================== 

class AENet(nn.Module):
	def __init__(self, arrDim):
		super(AENet, self).__init__()
		self.n_hidden_layers = len(arrDim) - 1

		# Encoding functions
		for i in range(self.n_hidden_layers) :
			f = nn.Linear(arrDim[i], arrDim[i+1])
			name = "fct" + str(i)
			setattr(self, name, f)

		# Decoding functions
		for i in range(self.n_hidden_layers) :
			f = nn.Linear(arrDim[-i-1], arrDim[-i-2])
			name = "fct" + str(i + self.n_hidden_layers)
			setattr(self, name, f)

	def forward(self, x):
		x = self.encode(x)
		x = self.decode(x)
		return x

	def encode(self, x):
		for i in range(self.n_hidden_layers - 1) :
			name = "fct" + str(i)
			fct = getattr(self, name)
			x = F.relu(fct(x))
		name = "fct" + str(self.n_hidden_layers - 1)
		fct = getattr(self, name)
		x = F.relu(fct(x))
		return x

	def decode(self, x):
		for i in range(self.n_hidden_layers - 1) :
			name = "fct" + str(i + self.n_hidden_layers)
			fct = getattr(self, name)
			x = F.relu(fct(x))
		name = "fct" + str(2*self.n_hidden_layers - 1)
		fct = getattr(self, name)
		x = fct(x)
		return x

# ===================================================================== 

class LinkNet(nn.Module):
	def __init__(self, arrDim):
		super(LinkNet, self).__init__()
		self.n_hidden_layers = len(arrDim) - 2

		for i in range( len(arrDim) - 1 ):
			f = nn.Linear( arrDim[i], arrDim[i+1] )
			name = "fct" + str(i)
			setattr(self, name, f)

	def forward(self, x):
		for i in range(self.n_hidden_layers):
			name = "fct" + str(i)
			fct = getattr(self, name)
			x = F.relu(fct(x))
		name = "fct" + str(self.n_hidden_layers)
		fct = getattr(self, name)
		x = F.relu(fct(x))
		return x

# =====================================================================

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

def getWeightedInputCodes(i, models, links, datasets, weights):
	codes = list()
	for j in range(len(models)):
		if i != j :
			code_externe = models[j].encode(datasets[j])
			code_interne = links[j][i](code_externe)*weights[j,i]
			codes.append(code_interne)

	code_moyen = ft.reduce(lambda x, y: x+y, codes)

	return code_moyen

# =====================================================================

def read_sparse(data_file_name):
	"""
	svm_read_problem(data_file_name) -> [m, y, x]
	Read LIBSVM-format data from data_file_name and return labels y
	and data instances x. m is a mapping of instance id to index
	in arrays x and y

	"""
	prob_y = []
	prob_x = []
	map_ii = {}
	i = 0
	dimData = 0
	for line in open(data_file_name):
		#print(line)
		if line[0] == "#":
			k = int(line.split("\t")[1].strip("\n"))
			map_ii[k] = i
			i += 1
		elif len(line) > 1:
			line = line.split(" ", 1)
		# In case an instance with all zero features
			if len(line) == 1: 
				prob_y += [int(line[0])]
				prob_x += [{0:0}]
				break
			label, features = line
			xi = {}
			for e in features.split(" "):
				ind, val = e.split(":")
				dimData = max(dimData, int(ind))
				xi[int(ind)] = float(val)
			prob_y += [int(label)]
			prob_x += [xi]

	dataset = torch.FloatTensor(i, dimData+1).zero_()
	for index, indiv in enumerate(prob_x):
		for key in indiv:
			dataset[index, key] = indiv[key]
	dataset = Variable(dataset)

	return (dataset, prob_y)

# =====================================================================

def read_sparse_to_pytorch(data_file_name):
	"""
	svm_read_problem(data_file_name) -> [m, y, x]
	Read LIBSVM-format data from data_file_name and return labels y
	and data instances x. m is a mapping of instance id to index
	in arrays x and y

	"""
	indexes = torch.LongTensor([[0],[0]])
	values = torch.FloatTensor([0])
	labels = []
	map_indexes = {}
	i = 0
	dimData = 0

	list_indexes = []
	list_values = []
	for line in open(data_file_name):
		if line[0] == "#":
			k = int(line.split("\t")[1].strip("\n"))
			map_indexes[k] = i
			i += 1
		elif len(line) > 1:
			line = line.split(" ", 1)
		# In case an instance with all zero features
			if len(line) == 1: 
				continue
			label, features = line
			labels += [label]
			features = features.split(" ")
			indexes_tmp = torch.LongTensor(2,len(features)).zero_()
			values_tmp = torch.FloatTensor(len(features)).zero_()
			for index_feature, e in enumerate(features):
				ind, val = e.split(":")
				dimData = max(dimData, int(ind))
				indexes_tmp[0,index_feature] = i-1
				indexes_tmp[1,index_feature] = int(ind)
				values_tmp[index_feature] = float(val)
			list_indexes.append(indexes_tmp)
			list_values.append(values_tmp)

	# indexes = ft.reduce(lambda x, y: torch.cat((x,y), 1), list_indexes)
	# values = ft.reduce(lambda x, y: torch.cat((x,y), 0), list_values)

	indexes = torch.cat(list_indexes, 1)
	values = torch.cat(list_values, 0)

	dataset = torch.sparse.FloatTensor(indexes, values, torch.Size([i,dimData+1]))

	return (dataset, labels)

# =====================================================================

def number_lines_file(file_name):
	with open(file_name) as f:
		for i, l in enumerate(f):
			pass
		return i+1

# =====================================================================

def learn_AENet(args):
		dataset = args["train"]
		dataset_test = args["test"]
		options = args["options"]
		id_net = args["id_net"]

		dimData = dataset.size()[1]

		print("View " + str(id_net) + " : learning...")

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
			if epoch % options["VERBOSE_STEP"] == 0 and options["VERBOSE"] : 
				print("Test Loss " + str(epoch) + " : " + str(loss.data[0]))

			# Stop when test error is increasing
			if loss.data[0] > min_test :
				test_fail += 1
				if test_fail > options["PATIENCE"] :
					if options["VERBOSE"] : print("Stop : test error increasing")
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
		print("View " + str(id_net) + " : done")
		print("\ttest loss : " + str(loss.data[0]))

		return net

# =====================================================================

def get_args_to_map_AE(train_datasets, test_datasets, options):
	f = lambda i : { "train" : train_datasets[i],"test" : test_datasets[i], "options" : options, "id_net" : i}
	return map(f, range(len(train_datasets)))

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
		print("Link " + str(i) + " ~ " + str(j))
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
			if epoch % options["VERBOSE_STEP"] == 0 and options["VERBOSE"]: 
				print("Test Loss " + str(epoch) + " : " + str(loss.data[0]))

			# Stop if test error is increasing
			if loss.data[0] > min_test :
				test_fail += 1
				if test_fail > options["PATIENCE"] :
					if options["VERBOSE"] : print("Stop : test error increasing")
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

		# print(net(data_test_in).data[1:10,:])
		# print(data_test_in.data[1:10,:])

		outputs = net(data_test_in)
		loss = criterion(outputs, data_test_out)
		print("\ttest loss : " + str(loss.data[0]))

		return net

# =====================================================================

def get_args_to_map_links(codes, codes_test, options):
	NVIEWS = len(codes)
	args = list()
	for i in range(NVIEWS):
		for j in range(NVIEWS):
			dic = {
				"id_in" : i,
				"id_out" : j,
				"data_in" : codes[i],
				"data_out" : codes[j],
				"test_in" : codes_test[i],
				"test_out" : codes_test[j],
				"options" : options	
			}
			args.append(dic)
	return args
# =====================================================================