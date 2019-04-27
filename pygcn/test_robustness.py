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

# Model and optimizer
model = torch.load('gcn_model.pth')

# Bounds
bound_calc = GCNBounds(model, features, adj, 0.01)
print(bound_calc)
torch.save({
            'lower_bound': bound_calc[0],
            'lower_bound': bound_calc[0],
            }, 'test_bounds.pt')