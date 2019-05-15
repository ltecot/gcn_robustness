from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy, compare_matricies
# from pygcn.models import GCN
from pygcn.robustness import GCNBoundsRelaxed, GCNBoundsFull
from pygcn.models import gcn_sequential_model

parser = argparse.ArgumentParser()
parser.add_argument('--relaxed',
            action = 'store_true',
            default = False)
parser.add_argument('--small',
            action = 'store_true',
            default = False)
parser.add_argument('--eps',
            default = 0.001,
            type = float)
parser.add_argument('--compare_file',
            default = "../../RecurJac-Develop/gcn_small_bound_matrices_eps1-100.pkl",
            type = str)

args = parser.parse_args()
relaxed = args.relaxed
small = args.small
eps = args.eps
targets = None  # TODO: add as arg.
# targets = [0, 2]

# print(relaxed, small, eps)

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
    bound_calc = GCNBoundsRelaxed(model, features, adj, eps, targets)
    LB = bound_calc.LB
    UB = bound_calc.UB
    # print("last upper: ", UB[-1])
    # print("last lower: ", LB[-1])
    # print("sums: ", torch.sum(bounds[0]), torch.sum(bounds[1]))
    for n in range(LB[-1].view(-1).shape[0]):
        print(str(LB[-1].view(-1).data[n]) + " < n_" + str(n) + " < " + str(UB[-1].view(-1).data[n]))
    pickle1 = pickle.load(open(args.compare_file, "rb"))
    compare_matricies(pickle1, {'LB': LB[-1].view(-1), 'UB': UB[-1].view(-1)})
    torch.save({
                'lower_bound': LB,
                'upper_bound': UB,
                }, 'test_bounds_relaxed.pt')
else: # full
    bound_calc = GCNBoundsFull(model, features, adj, eps, targets)
    LB = bound_calc.LB
    UB = bound_calc.UB
    # print("last upper: ", UB[-1].view(-1))
    # print("last lower: ", LB[-1].view(-1))
    # print("sums: ", torch.sum(bounds[0]), torch.sum(bounds[1]))
    for n in range(LB[-1].view(-1).shape[0]):
        print(str(LB[-1].view(-1).data[n]) + " < n_" + str(n) + " < " + str(UB[-1].view(-1).data[n]))
    pickle1 = pickle.load(open(args.compare_file, "rb"))
    compare_matricies(pickle1, {'LB': LB[-1].view(-1), 'UB': UB[-1].view(-1)})
    torch.save({
                'lower_bound': LB,
                'upper_bound': UB,
                }, 'test_bounds_full.pt')
    # Debug
    torch.set_printoptions(profile="full")

    # 463, 466
    # print("UB: ", UB[-2].view(-1))
    # print("LB: ", LB[-2].view(-1))

    # pickle1 = pickle.load(open("../../RecurJac-Develop/gcn_small_bound_firstlayer_eps1-100.pkl", "rb"))
    # compare_matricies(pickle1, {'LB': LB[-2].view(-1), 'UB': UB[-2].view(-1)})
    # pickle1 = pickle.load(open("../../RecurJac-Develop/gcn_small_bound_generalchecks_eps1-100.pkl", "rb"))
    # gc = {'upper_k': bound_calc.alpha_u[-1].view(-1), 
    #     'lower_k': bound_calc.alpha_l[-1].view(-1), 
    #     'upper_b': bound_calc.beta_u[-1].view(-1), 
    #     'lower_b': bound_calc.beta_l[-1].view(-1), 
    #     'diags_ub': bound_calc.lmd[-1][:, -1], 
    #     'diags_lb': bound_calc.omg[-1][:, -1], 
    #     'l_ub': bound_calc.delta[-2][:, -1], 
    #     'l_lb': bound_calc.theta[-2][:, -1], 
    #     'A_UB': bound_calc.Lambda[-3].t(), 
    #     'A_LB': bound_calc.Omega[-3].t(), 
    # }
    # compare_matricies(pickle1, gc)
    # pickle1 = pickle.load(open("../../RecurJac-Develop/gcn_small_bound_boundcheck_eps1-100.pkl", "rb"))
    # pickle2 = pickle.load(open("gcn_small_bound_boundcheck_eps1-100.pkl", "rb"))
    # compare_matricies(pickle1, pickle2)

    # print("UB shape: ", UB[-2].shape)
    # print("UB: ", UB[-2].view(-1)[556])
    # print("LB: ", LB[-2].view(-1)[556])
    # print("beta u: ", bound_calc.beta_u[-1].view(-1)[556])
    # print("beta l: ", bound_calc.beta_l[-1].view(-1)[556])
    # print("alpha u: ", bound_calc.alpha_u[-1].view(-1)[556])
    # print("alpha l: ", bound_calc.alpha_l[-1].view(-1)[556])
    # print("delta shape: ", bound_calc.delta[-2].shape)
    # print("theta shape: ", bound_calc.theta[-2].shape)
    # print("delta: ", bound_calc.delta[-2][:, 463].view(-1).nonzero())
    # print("theta: ", bound_calc.theta[-2][:, 463].view(-1))

    # print("Lambda: ", bound_calc.Lambda[0].shape)
    # print("Lambda: ", bound_calc.Lambda[0][0])
    # print("alpha length: ", len(bound_calc.alpha_u))
    # print("first alpha upper: ", bound_calc.alpha_u[0].view(-1))
    # print("first alpha lower: ", bound_calc.alpha_l[0].view(-1))
    # print(bound_calc.lmd[1].shape)
    # print("lmd shape: ", bound_calc.lmd[1].shape)
    # print("omg shape: ", bound_calc.omg[1].shape)
    # print("lmd: ", bound_calc.lmd[1][:, 0].view(-1).data)
    # print("omg: ", bound_calc.omg[1][:, 0].view(-1).data)
    # print("first upper: ", UB[0].view(-1))
    # print("first lower: ", LB[0].view(-1))
    # lyr_l = bound_calc.l[2].view(-1)
    # lyr_u = bound_calc.u[2].view(-1)
    # for n in range(lyr_l.shape[0]):
    #     print(str(lyr_l.data[n]) + " < n_" + str(n) + " < " + str(lyr_u.data[n]))
    # # print("pre_ac up range: ", bound_calc.u[1].data.view(-1))
    # # print("pre_ac low range: ", bound_calc.l[1].data.view(-1))
    # print(model.forward(features).data.view(-1))
    # for param in model.parameters():
    #     print(param)
    # print("pre_ac up range: ", bound_calc.u[1].data.view(-1)[:10])
    # print("pre_ac low range: ", bound_calc.l[1].data.view(-1)[:10])
    # print("pre_ac up range: ", bound_calc.u[1].data.size())
    
