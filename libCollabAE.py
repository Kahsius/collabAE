import torch
import torch.nn as nn
import torch.nn.functional as F
import functools as ft
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

def svm_read_problem(data_file_name):
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
	for line in open(data_file_name):
		#print(line)
		if line[0] == "#":
			print(line)
			print(int(line.split("\t")[1].strip("\n")))
			map_ii[int(line.split("\t")[1].strip("\n"))] = i
			line = [line]
		elif len(line) > 1:
			print(line)
			line = line.split(" ", 1)
		# In case an instance with all zero features
		print(len(line))
		if len(line) == 1: line += ['']
		label, features = line
		print("OK")
		xi = {}
		print(features.split(" "))
		for e in features.split(" "):
			ind, val = e.split(":")
			xi[int(ind)] = float(val)
		prob_y += [float(label)]
		prob_x += [xi]
		i += 1

	return (map_ii,prob_y, prob_x)
