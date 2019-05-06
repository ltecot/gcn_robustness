# File to convert the GCN model to a linear one and save it.

# Command for converter script in Recur-jac repo:
# From convert folder:
# cp ../../gcn_robustness/pygcn/linear_gcn_model_small.pth .
# python torch2keras.py -i linear_gcn_model_small.pth -o linear_gcn_model_small.h5 143300 1600 700

import torch

from pygcn.models import convert_gcn_to_feedforward

model = torch.load('gcn_model_small.pth')
linear_model = convert_gcn_to_feedforward(model)
for layer in linear_model:
    if isinstance(layer, torch.nn.Linear):
        print(layer.weight.shape)
torch.save(linear_model, "linear_gcn_model_small.pth")