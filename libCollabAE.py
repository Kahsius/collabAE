import torch
import torch.nn as nn
import torch.nn.functional as F

class AENet(nn.Module):
	def __init__(self, arrDim):
		super(AENet, self).__init__()
		self.n_hidden_layers = len(arrDim) - 1
		self.layers = list()

		# Encoding functions
		for i in range(self.n_hidden_layers) :
			f = nn.Linear(arrDim[i], arrDim[i+1])
			self.layers.append(f)

		# Decoding functions
		for i in range(self.n_hidden_layers) :
			f = nn.Linear(arrDim[-i-1], arrDim[-i-2])
			self.layers.append(f)

	def forward(self, x):
		x = self.encode(x)
		x = self.decode(x)
		return x

	def encode(self, x):
		for i in range(self.n_hidden_layers) :
			x = F.relu(self.layers[i](x))
		return x

	def decode(self, x):
		for i in range(self.n_hidden_layers - 1) :
			x = F.relu(self.layers[n_hidden_layers + i](x))
		x = self.layers[-1](x)
		return x

