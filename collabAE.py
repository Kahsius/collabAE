import torch.optim as optim
import pandas as pd
import numpy as np
import sys
import functools as ft
from sklearn.preprocessing import scale
from libCollabAE import *

# PARAMETERS
VERBOSE = False

# HYPERPARAMETERS
PTEST = .1
NSTEPS = 5000
NVIEWS = 3
LAYERS_AE = [30, 10]
LAYERS_LINKS = [30, 30]
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
print("\nnData : " + str(nData))
print("nTest : " + str(nTest))

# TRAIN AND TEST SETS
test_data = Variable(torch.from_numpy(data[:nTest,:]).float())
train_data = Variable(torch.from_numpy(data[nTest:,:]).float())

indexes = getIndexesViews(dimData, NVIEWS)
print("Indexes : " + str(indexes))
train_datasets = getViewsFromIndexes(train_data, indexes)
test_datasets = getViewsFromIndexes(test_data, indexes)

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
for i in range(NVIEWS):
	codes = list()
	for j in range(NVIEWS):
		if i != j :
			code_externe = models[j].encode(test_datasets[j])
			code_interne = links[j][i](code_externe)
			codes.append(code_interne)

	code_moyen = ft.reduce(lambda x, y: (x.data+y.data)/(NVIEWS-1), codes)
	code_moyen = Variable(code_moyen)

	indiv_reconstruit = models[i].decode(code_moyen)
	criterion = nn.MSELoss()
	loss = criterion(indiv_reconstruit, test_datasets[i])

	print("Reconstruction error view " + str(i) + " : " + str(loss.data[0]))