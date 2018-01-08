import torch.optim as optim
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import scale
from libCollabAE import *

torch.manual_seed(123)

# PARAMETERS
VERBOSE = False

# HYPERPARAMETERS
PTEST = .1
NSTEPS = 5000
NSTEPS_WEIGHTS = 300
NVIEWS = 3
LAYERS_AE = [10]
LAYERS_LINKS = [50]
LEARNING_RATE_AE = 0.03
LEARNING_RATE_LINKS = 0.06

# DATA PREPROCESSING
data = pd.read_csv("data/wdbc.data", header=None).values[:,2:]
data = np.array(data, dtype='float')
data = scale(data)
np.random.shuffle(data)

# DATA INFORMATIONS
dimData = data.shape[1]
nData = data.shape[0]
nTest = int(nData * PTEST)

# TRAIN AND TEST SETS
test_data = Variable(torch.from_numpy(data[:nTest,:]).float())
train_data = Variable(torch.from_numpy(data[nTest:,:]).float())

indexes = getIndexesViews(dimData, NVIEWS)
train_datasets = getViewsFromIndexes(train_data, indexes)
test_datasets = getViewsFromIndexes(test_data, indexes)

print("\n")
print("nData : " + str(nData))
print("Test : " + str(nTest))
print("dim AE : " + str([dimData] + LAYERS_AE))
print("dim Links : " + str([LAYERS_AE[-1]] + LAYERS_LINKS + [LAYERS_AE[-1]]))
print("Indexes : " + str(indexes))
print("\n")

# LEARNING ALL THE MODELS AND GET THE CODES
models = list()
codes = list()
codes_test = list()

for i in range(NVIEWS):
	print("Learning view " + str(i))
	dataset = train_datasets[i]
	dataset_test = test_datasets[i]
	dimData = indexes[i+1] - indexes[i]

	# MODEL DEFINITION
	net = AENet( [dimData] + LAYERS_AE )
	criterion = nn.MSELoss()
	optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE_AE)

	# LEARNING
	for epoch in range(NSTEPS):

		if VERBOSE:
			# Test information
			outputs = net(dataset_test)
			loss = criterion(outputs, dataset_test)
			if epoch % 100 == 0 : 
				print("Test Loss " + str(epoch) + " : " + str(loss.data[0]))
		
		optimizer.zero_grad()
		
		# Train information
		outputs = net(dataset)
		loss = criterion(outputs, dataset)

		# Parameters optimization
		loss.backward()
		optimizer.step()

	outputs = net(dataset_test)
	loss = criterion(outputs, dataset_test)
	print("\ttest loss : " + str(loss.data[0]))

	models.append(net)
	print("Encoding training and test dataset...\n")
	# Codes gathering
	code = net.encode(dataset)
	code = Variable(code.data, requires_grad = False)
	codes.append(code)

	code_test = net.encode(dataset_test)
	code_test = Variable(code_test.data, requires_grad = False)
	codes_test.append(code_test)

# LEARNING OF THE LINKS
links = list()
for i in range(NVIEWS):
	links.append(list())

for i in range(NVIEWS):
	for j in range(NVIEWS):
		if i == j :
			links[i].append([])
		else :

			# GET THE CODE TO LINK
			data_in = codes[i]
			data_out = codes[j]

			data_test_in = codes_test[i]
			data_test_out = codes_test[j]

			dimData_in = data_in.size()[1]
			dimData_out = data_out.size()[1]

			# DEFINE THE MODEL
			print("Link " + str(i) + " ~ " + str(j))
			net = LinkNet( [dimData_in] + LAYERS_LINKS + [dimData_out] )
			criterion = nn.MSELoss()
			optimizer = optim.SGD( net.parameters(), lr=LEARNING_RATE_LINKS)

			# LEARNING
			for epoch in range(NSTEPS):
				# Test information
				if VERBOSE:
					outputs = net(data_test_in)
					loss = criterion(outputs, data_test_out)
					if epoch % 100 == 0 : 
						print("Test Loss " + str(epoch) + " : " + str(loss.data[0]))

				optimizer.zero_grad()
				
				# Train information
				outputs = net(data_in)
				loss = criterion(outputs, data_out)

				# Parameters optimization
				loss.backward()
				optimizer.step()

			# print(net(data_test_in).data[1:10,:])
			# print(data_test_in.data[1:10,:])

			outputs = net(data_test_in)
			loss = criterion(outputs, data_test_out)
			print("\ttest loss : " + str(loss.data[0]))

			links[i].append(net)

print("\n")

# TESTING THE RECONSTRUCTION
# PROTO WEIGTHING WITH GRAD
w = torch.FloatTensor(NVIEWS,NVIEWS).zero_()+1/(NVIEWS-1)
weights = (Variable(w, requires_grad=True))
optimizer = optim.SGD([weights], lr=LEARNING_RATE_AE)
criterion = nn.MSELoss()

for i in range(NVIEWS):
	for epoch in range(NSTEPS_WEIGHTS):
		optimizer.zero_grad()

		code_moyen = getWeightedInputCodes(i, models, links, train_datasets, weights)
		# code_moyen = ft.reduce(lambda x, y: (x.data+y.data)/(NVIEWS-1), codes)
		indiv_reconstruit = models[i].decode(code_moyen)
		
		loss = criterion(indiv_reconstruit, train_datasets[i])
		loss.backward()
		optimizer.step()

		if epoch % 100 == 0 and VERBOSE:
			code_test_moyen = getWeightedInputCodes(i, models, links, test_datasets, weights)
			indiv_reconstruit = models[i].decode(code_test_moyen)
			loss = criterion(indiv_reconstruit, test_datasets[i])
			print("Reconstruction error view " + str(i) + " : " + str(loss.data[0]))

	code_test_moyen = getWeightedInputCodes(i, models, links, test_datasets, weights)
	indiv_reconstruit = models[i].decode(code_test_moyen)
	loss = criterion(indiv_reconstruit, test_datasets[i])
	print("Reconstruction error view " + str(i) + " : " + str(loss.data[0]))
print("\n")

print("Weights")
print(weights[:,:])