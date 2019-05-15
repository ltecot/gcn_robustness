import torch.nn as nn
import torch.nn.functional as F

from pygcn.layers import GraphConvolution
from pygcn.utils import kronecker

def gcn_sequential_model(nfeat, nhid, nclass, adj):
    model = nn.Sequential(
        GraphConvolution(nfeat, nhid, adj, activation=F.relu),
        GraphConvolution(nhid, nclass, adj, activation=None)
    )
    return model

# Take in a gcn model above and convert it to a sequential model
def convert_gcn_to_feedforward(model):
    modules = []
    for layer in model:
        # print(layer)
        # print(isinstance(layer, GraphConvolution))
        if isinstance(layer, GraphConvolution):
            tensor_weight = kronecker(layer.adj.t().contiguous(), layer.weight)
            new_layer = nn.Linear(tensor_weight.shape[0], tensor_weight.shape[1])
            # new_layer = nn.Linear(tensor_weight.shape[1], tensor_weight.shape[0])
            # print(new_layer.weight.shape)
            # print(tensor_weight.shape)
            new_layer.weight.data = tensor_weight.t().data
            new_layer.bias.data.fill_(0)
            modules.append(new_layer)
            modules.append(nn.ReLU())
        else:
            raise ValueError("Only GCN layers supported.")
    modules = modules[:-1]
    sequential = nn.Sequential(*modules)
    return sequential