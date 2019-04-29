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
from pygcn.robustness import GCNBounds

np.random.seed(42)
torch.manual_seed(42)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()
adj = adj.to_dense()  # Temporarily to have less headaches. Also note that each layer must have an adj, which is an issue.

# Model and optimizer
model = torch.load('gcn_model.pth')

# Bounds
bound_calc = GCNBounds(model, features, adj, 0.01)
bounds = bound_calc.bounds
print(bounds)
torch.save({
            'lower_bound': bounds[0],
            'upper_bound': bounds[1],
            }, 'test_bounds.pt')