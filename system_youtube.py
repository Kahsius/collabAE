from collabAE import *
import os

path = "data/dir_data/test/"
filenames = os.listdir(path)
filenames = [f for f in filenames if os.path.isfile(os.path.join(path, f))]
PTEST = .1
nTest = int(11930 * PTEST)

options = {
	"VERBOSE" : True,
	"VERBOSE_STEP" : 20,
	"NSTEPS" : 1000,
	"NSTEPS_WEIGHTS" : 200,
	"LAYERS_AE" : [1000, 200, 30],
	"LAYERS_LINKS" : [100],
	"LEARNING_RATE_AE" : 0.01,
	"LEARNING_RATE_LINKS" : 0.03,
	"LEARNING_RATE_WEIGHTS" : 0.01,
	"MOMENTUM" : 0.9,
	"PATIENCE" : 100
}

train_datasets = []
test_datasets = []
names = []

for filename in filenames :
	short_name = filename.split('.')[0]
	dataset, labels = read_sparse_to_pytorch(path + filename)
	if dataset.size()[0] == 11930 :
		print("Using " + filename)
		names.append(short_name)
		dataset = Variable(dataset.to_dense())
		train_datasets.append(dataset[nTest:,:])
		test_datasets.append(dataset[:nTest,:])
		if len(train_datasets) == 3 : break

learnCollabSystem(train_datasets, test_datasets, options)