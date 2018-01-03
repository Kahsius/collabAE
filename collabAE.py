from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from libCollabAE import *

PTEST = .1
NSTEPS = 100

data = pd.read_csv("data/wdbc.data", header=None).values[:,2:]
data = np.array(data, dtype='float')
data = scale(data)
np.random.shuffle(data)

dimData = data.shape[1]
nData = data.shape[0]
nTest = int(nData * PTEST)
print("nData : " + str(nData))
print("nTest : " + str(nTest))

test_data = Variable(torch.from_numpy(data[:nTest,:]).float())
train_data = Variable(torch.from_numpy(data[nTest:,:]).float())

aenet = AENet([dimData, 30, 10])
criterion = nn.MSELoss()
optimizer = optim.SGD(aenet.parameters(), lr=0.03)

for epoch in range(NSTEPS):
	optimizer.zero_grad()
	outputs = aenet(test_data)
	loss = criterion(outputs, test_data)
	if epoch % 100 == 0 : print("Loss " + str(epoch) + " : " + str(loss.data[0]))
	outputs = aenet(train_data)
	loss = criterion(outputs, train_data)
	loss.backward()
	optimizer.step()