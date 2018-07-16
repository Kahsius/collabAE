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
        #x = self.normalize(x)
        for i in range(self.n_hidden_layers - 1) :
            name = "fct" + str(i)
            fct = getattr(self, name)
            x = F.relu(fct(x))
        name = "fct" + str(self.n_hidden_layers - 1)
        fct = getattr(self, name)
        
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
    def __init__(self, arrDim, clampOutput = True):
        super(LinkNet, self).__init__()
        self.n_hidden_layers = len(arrDim) - 2
        self.clampOutput = clampOutput

        for i in range( len(arrDim) - 1 ):
            f = nn.Linear( arrDim[i], arrDim[i+1] )
            name = "fct" + str(i)
            setattr(self, name, f)

    def forward(self, x):
        for i in range(self.n_hidden_layers+1):
            name = "fct" + str(i)
            fct = getattr(self, name)
            x = fct(x)
            if self.clampOutput :
                x = F.relu(x)
                ones = Variable(torch.ones(x.shape))
                x = torch.min(x, ones)
        return x

# ===================================================================== 

class CollabSystem4():
    def __init__(self, autoencoders, links, weights):
        self.ae = autoencoders
        self.links = links
        self.w = weights

    def forward(self, id_view, datasets):
        codes = list()
        for i in range(len(datasets)):
            codes.append(self.ae[i].encode(datasets[i]))
        indiv_reconstruit = get_weighted_outputs(id_view, codes, self.links, self.w[id_view])
        return indiv_reconstruit

# =====================================================================
