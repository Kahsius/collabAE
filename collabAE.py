import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

PTEST = .1

data = pd.read_csv("data/wdbc.data", header=None).values[:,2:]
data = np.array(data, dtype='float')
np.random.shuffle(data)

dimData = data.shape[1]
nData = data.shape[0]
nTest = int(nData * PTEST)
print("nData : " + str(nData))
print("nTest : " + str(nTest))

test_data = Variable(torch.from_numpy(data[:nTest,:]).float())
train_data = Variable(torch.from_numpy(data[nTest:,:]).float())

class AENet(nn.Module):
	def __init__(self, dimData):
		super(AENet, self).__init__()
		self.fc1 = nn.Linear(dimData, 30)
		self.fc2 = nn.Linear(30, 10)
		self.fc3 = nn.Linear(10, 30)
		self.fc4 = nn.Linear(30, dimData)

	def forward(self, x):
		x = self.encode(x)
		x = self.decode(x)
		return x

	def encode(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return x

	def decode(self, x):
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x

aenet = AENet(dimData)
criterion = nn.MSELoss()
optimizer = optim.SGD(aenet.parameters(), lr=0.0001)

for epoch in range(1000):
	optimizer.zero_grad()
	outputs = aenet(train_data)
	loss = criterion(outputs, train_data)
	loss.backward()
	optimizer.step()
	# print("Loss " + str(epoch) + " : " + str(loss.data[0]))

print(str(train_data.data.numpy()))
print(str(aenet(train_data).data.numpy()))