from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
# from pygcn.models import GCN
from pygcn.robustness import GCNBoundsRelaxed, GCNBoundsFull
from pygcn.models import gcn_sequential_model

# settings
relaxed = False
small = True
# eps = 0.001
eps = 0

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()
adj = adj.to_dense()  # Temporarily to have less headaches. Also note that each layer must have an adj, which is an issue.

# print(adj)

# Model and optimizer
# model = torch.load('gcn_model.pth')
# model = torch.load('gcn_model_small.pth')

if small:
    adj = adj[0:100, 0:100]
    features = features[:100]
    labels = labels[:100]
    model = gcn_sequential_model(nfeat=features.shape[1],
                             nhid=16,  # From train.py args 
                             nclass=labels.max().item() + 1,
                             adj=adj)
    model.load_state_dict(torch.load('gcn_model_small.pth'))
else:
    model = gcn_sequential_model(nfeat=features.shape[1],
                             nhid=16,  # From train.py args 
                             nclass=labels.max().item() + 1,
                             adj=adj)
    model.load_state_dict(torch.load('gcn_model_small.pth'))

# Bounds
if relaxed:
    bound_calc = GCNBoundsRelaxed(model, features, adj, eps)
    LB = bound_calc.LB
    UB = bound_calc.UB
    print("last upper: ", UB[-1])
    print("last lower: ", LB[-1])
    # print("sums: ", torch.sum(bounds[0]), torch.sum(bounds[1]))
    torch.save({
                'lower_bound': LB,
                'upper_bound': UB,
                }, 'test_bounds_relaxed.pt')
else: # full
    bound_calc = GCNBoundsFull(model, features, adj, eps)
    LB = bound_calc.LB
    UB = bound_calc.UB
    print("last upper: ", UB[-1].view(-1))
    print("last lower: ", LB[-1].view(-1))
    # print("sums: ", torch.sum(bounds[0]), torch.sum(bounds[1]))
    torch.save({
                'lower_bound': LB,
                'upper_bound': UB,
                }, 'test_bounds_full.pt')
    # Debug
    torch.set_printoptions(profile="full")
    # print("first upper: ", UB[0])
    # print("first lower: ", LB[0])
    # lyr_l = bound_calc.l[2].view(-1)
    # lyr_u = bound_calc.u[2].view(-1)
    # for n in range(lyr_l.shape[0]):
    #     print(str(lyr_l.data[n]) + " < n_" + str(n) + " < " + str(lyr_u.data[n]))
    # # print("pre_ac up range: ", bound_calc.u[1].data.view(-1))
    # # print("pre_ac low range: ", bound_calc.l[1].data.view(-1))
    print(model.forward(features).data.view(-1))
    # print("pre_ac up range: ", bound_calc.u[1].data.view(-1)[:10])
    # print("pre_ac low range: ", bound_calc.l[1].data.view(-1)[:10])
    # print("pre_ac up range: ", bound_calc.u[1].data.size())
    
