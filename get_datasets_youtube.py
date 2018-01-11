from collabAE import *
import sys
import pickle
import os

filenames = os.listdir("data/dir_data/test")
filenames = [f for f in filenames if os.path.isfile(os.path.join("data/dir_data/test/", f))]
# PTEST = .1

# options = {
# 	"VERBOSE" : True,
# 	"VERBOSE_STEP" : 20,
# 	"NSTEPS" : NSTEPS,
# 	"NSTEPS_WEIGHTS" : NSTEPS_WEIGHTS,
# 	"LAYERS_AE" : LAYERS_AE,
# 	"LAYERS_LINKS" : LAYERS_LINKS,
# 	"LEARNING_RATE_AE" : LEARNING_RATE_AE,
# 	"LEARNING_RATE_LINKS" : LEARNING_RATE_WEIGHTS,
# 	"LEARNING_RATE_WEIGHTS" : LEARNING_RATE_WEIGHTS,
# 	"MOMENTUM" : MOMENTUM,
# 	"PATIENCE" : PATIENCE
# }

# train_datasets = list()
# test_datasets = list()
# for filename in filenames :
# 	print("Loading " + filename)
# 	dataset, labels = read_sparse(filename)
# 	nData = dataset.size()[0]
# 	nTest = int(nData * PTEST)
# 	train_datasets.append(dataset[nTest:,:])
# 	test_datasets.append(dataset[:nTest,:])

path = "data/dir_data/test/"
for filename in filenames :
	print("Loading " + filename)
	print(number_lines_file(path + filename))
	dataset, labels = read_sparse_to_pytorch(path + filename)
	print(dataset.size())
	# short_name = filename.split('.')[0]
	# with open(path + "pickle/" + short_name + ".pkl", "wb") as f:
	# 	print(dataset.size())
	# 	pickle.dump(dataset, f)

# with open('data/data_test_youtube.pkl', 'wb') as f:
# 	pickle.dump([datasets, labels], f)

print("Datasets saved")

#learnCollabSystem(train_datasets, test_datasets, options)