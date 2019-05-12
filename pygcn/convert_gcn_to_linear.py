# File to convert the GCN model to a linear one and save it.

# Command for converter script in Recur-jac repo:
# From convert folder:
# cp ../../gcn_robustness/pygcn/linear_gcn_model_small.pth .
# python torch2keras.py -i linear_gcn_model_small.pth -o linear_gcn_model_small.h5 143300 1600 700

import torch
import torch.nn as nn

from pygcn.utils import load_data
from pygcn.models import convert_gcn_to_feedforward, gcn_sequential_model

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()
adj = adj.to_dense()  # Temporarily to have less headaches. Also note that each layer must have an adj, which is an issue.
adj = adj[0:100, 0:100]
features = features[:100]
labels = labels[:100]
model = gcn_sequential_model(nfeat=features.shape[1],
                             nhid=16,  # From train.py args 
                             nclass=labels.max().item() + 1,
                             adj=adj)
model.load_state_dict(torch.load('gcn_model_small.pth'))
linear_model = convert_gcn_to_feedforward(model)
for layer in linear_model:
    if isinstance(layer, nn.Linear):
        print(layer)
torch.save(linear_model.state_dict(), "linear_gcn_model_small.pth")
print(linear_model.forward(features.view(1, -1)).data.view(-1))
# torch.save(linear_model, "linear_gcn_model_small.pth")
# print(linear_model)
# for k, elem in linear_model.state_dict().items():
#     print(k, elem.shape)