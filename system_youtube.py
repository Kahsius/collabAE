from collabAE import *
import os

path = "data/dir_data/test/"
filenames = os.listdir(path)
filenames = [f for f in filenames if os.path.isfile(os.path.join(path, f))]
PTEST = .1
NINDIV = 3000
NFILES = 3
nTest = int(11930 * PTEST)

train_datasets = []
test_datasets = []
names = []

print("\nFiles used :")
for filename in filenames :
	short_name = filename.split('.')[0]
	if short_name.find("vision") != -1:
		dataset, labels = read_sparse_to_pytorch(path + filename)
		if dataset.size()[0] == 11930 :
			print("\t- " + filename)
			names.append(short_name)
			dataset = Variable(dataset.to_dense())
			dataset = dataset[:NINDIV,:]
			labels = labels[:NINDIV]
			train_datasets.append(dataset[nTest:,:])
			test_datasets.append(dataset[:nTest,:])
			train_labels = Variable(torch.LongTensor(labels[nTest:]))
			test_labels = Variable(torch.LongTensor(labels[:nTest]))
			if len(train_datasets) == 3 : break

options = {
	"VERBOSE" : True,
	"VERBOSE_STEP" : 20,
	"NSTEPS" : 1000,
	"NSTEPS_WEIGHTS" : 200,
	"LAYERS_AE" : [1000, 200, 30],
	"LAYERS_LINKS" : [300, 100],
	"LAYERS_CLASSIF" : [300, 100],
	"LEARNING_RATE_AE" : 0.01,
	"LEARNING_RATE_LINKS" : 0.03,
	"LEARNING_RATE_WEIGHTS" : 0.01,
	"LEARNING_RATE_CLASSIF" : 0.001,
	"MOMENTUM" : 0.9,
	"PATIENCE" : 100,
	"train_labels" : train_labels,
	"test_labels" : test_labels 
}

print("NINDIV :" + str(NINDIV))
print("\twith nTest : " + str(nTest))
print("LAYERS_AE : " + str(options["LAYERS_AE"]))
print("LAYERS_LINKS : " + str(options["LAYERS_LINKS"]))
print("LAYERS_CLASSIF : " + str(options["LAYERS_CLASSIF"]))
print("\n")

learnCollabSystem3(train_datasets, test_datasets, options)