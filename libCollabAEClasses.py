import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from libCollabAEUtils import *

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
        
        # x = F.relu(fct(x))
        # x = F.softmax(fct(x), dim=1)
        x = F.relu(fct(x))
        ones = Variable(torch.ones(x.shape))
        x = torch.min(x, ones)
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
        for i in range(self.n_hidden_layers+1):
            name = "fct" + str(i)
            fct = getattr(self, name)
            x = F.relu(fct(x))
            ones = Variable(torch.ones(x.shape))
            x = torch.min(x, ones)
        return x

# ===================================================================== 

class ClassifNet(nn.Module):
    def __init__(self, arrDim):
        super(ClassifNet, self).__init__()
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
        x = F.softmax(fct(x), dim=1)
        return x

    def getClasses(self, x):
        r = self.forward(x)
        _, predictions = torch.max(r, 1)
        return predictions

# ===================================================================== 

class CollabSystem():
    def __init__(self, autoencoders, links, weights):
        self.ae = autoencoders
        self.links = links
        self.w = weights

    def forward(self, id_view, datasets):
        code_moyen = getWeightedInputCodes3(id_view, datasets, self.links, self.w[id_view])
        indiv_reconstruit = self.ae[id_view].decode(code_moyen)
        return indiv_reconstruit

# =====================================================================