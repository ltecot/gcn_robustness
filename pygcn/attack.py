from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import gcn_sequential_model

class PGD(object):
    def __init__(self,model):
        self.model = model

    def get_loss(self,features,labels, idx_test, TARGETED):
        output = self.model(features)
        #print(output, label_or_target)
        loss_test = F.nll_loss(F.log_softmax(output[idx_test], dim=1), labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        #print(loss)
        #print(c.size(),modifier.size())
        return loss_test, acc_test

    def pgd(self, features, labels, idx_test, mask, epsilon, eta, TARGETED=False):
        x_in = features
        yi = Variable(labels)
        x_adv = Variable(features, requires_grad=True)
        for it in range(10):
            error, acc = self.get_loss(x_adv,yi, idx_test, TARGETED)
            if (it)%1==0:
                print(error.data.item()) 
            #x_adv.grad.data.zero_()
            error.backward(retain_graph=True)
            #print(gradient)
            #print(x_adv.grad.size())
            masked_grad= x_adv.grad*mask
            masked_grad.sign_()
            if TARGETED:
                x_adv.data = x_adv.data - eta* masked_grad
            else:
                x_adv.data = x_adv.data + eta* masked_grad
            diff = x_adv.data - x_in
            diff.clamp_(-epsilon,epsilon)
            x_adv.data=(diff + x_in).clamp_(0, 1)
        return x_adv

    def __call__(self, features, labels, idx_test, mask, epsilon, eta=0.1, TARGETED=False):
        adv = self.pgd(features, labels, idx_test, mask, epsilon, eta, TARGETED)
        return adv  



parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cora",
                    help='Dataset to be used')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)


model = gcn_sequential_model(nfeat=features.shape[1],
                             nhid=args.hidden, 
                             nclass=labels.max().item() + 1,
                             adj=adj)

model = torch.load("gcn_model.pth")
perm =  torch.randperm(idx_test.size()[0])
idx = perm[:1]
test_samples = idx_test[idx]

k= 10
perm =  torch.randperm(idx_train.size()[0])
idx = perm[:k]
modified_nodes_idx = idx_train[idx]
mask = torch.FloatTensor(features.size())
mask.zero_()
mask[modified_nodes_idx,:] = 1
attack = PGD(model)
epsilon = 0.3
features_adv = attack(features, labels, test_samples, mask, epsilon)

