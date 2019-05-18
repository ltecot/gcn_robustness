from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import h5py
from utils import load_data, accuracy, elision_error
from models import gcn_sequential_model
from node_masking import select_target_node, select_perturb_node
from robustness import GCNBoundsTwoLayer

def test(model, features, labels, idx_test):
    #model.eval()
    # output = model(features, adj)
    output = model(features)
    #print(output)
    #print(labels)
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    loss_test = F.nll_loss(F.log_softmax(output[idx_test], dim=1), labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
   
    preds = output[idx_test].max(1)[1].type_as(labels[idx_test])
    #print(output[idx_test].size(), len(idx_test))
    return preds, acc_test 

def evaluate(data):
    total_loss = 0
    total_acc  = 0
    total_pred = []
    total_labs = []

    t_test = time()
    N = len(data)

    for start in range(0, N, FLAGS.test_batch_size):
        end = min(start+FLAGS.test_batch_size, N)
        batch = data[start:end]
        feed_dict = eval_sch.batch(batch)

        los, acc, prd = test_model.run_one_step(sess, feed_dict)
        batch_size = prd.shape[0]
        total_loss += los * batch_size
        total_acc  += acc * batch_size
        total_pred.append(prd)
        total_labs.append(feed_dict[placeholders['labels']])

    total_loss /= N
    total_acc  /= N
    total_pred = np.vstack(total_pred)
    total_labs = np.vstack(total_labs)

    micro, macro = calc_f1(total_pred, total_labs, multitask)
    return total_loss, total_acc, micro, macro, (time()-t_test)


def Test():
    # Testing
    test_cost, test_acc, micro, macro, test_duration = evaluate(test_d)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc),
          "mi F1={:.5f} ma F1={:.5f} ".format(micro, macro),
          "time=", "{:.5f}".format(test_duration))
    remaining = np.array(list(set(range(num_data)) - set(test_d)), dtype=np.int32)



class PGD(object):
    def __init__(self,model):
        self.model = model

    def get_loss(self,features,labels, idx_test, TARGETED):
        output = self.model(features)
        #print(output, label_or_target)
        loss_test = F.nll_loss(F.log_softmax(output[idx_test], dim=1), labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        #print(output[idx_test])
        #print(loss)
        #print(c.size(),modifier.size())
        return loss_test, acc_test

    def pgd(self, features, labels, idx_test, mask, epsilon, eta, TARGETED=False):
        x_in = features
        yi = Variable(labels)
        x_adv = Variable(features, requires_grad=True)
        for it in range(10):
            error, acc = self.get_loss(x_adv,yi, idx_test, TARGETED)
            #if (it)%1==0:
            #    print(error.data.item(), acc.data.item()) 
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

parser.add_argument('--n_neigh', type=int, default=0,
                    help='number of neighbors of target node')

parser.add_argument('--hops', type=int, default=1,
                    help='hops of neighbors of target node')
parser.add_argument('--epsilon', type=float, default=0.01,
                    help='epsilon in the PGD attack')

parser.add_argument('--batch_size', type=int, default=10,
                    help='batch_size in the PGD attack')

parser.add_argument('--verbose', action='store_true', default=False,
                    help='verbose.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
if args.dataset == "reddit":
    num_data, train_adj, full_adj, features, train_features, test_features, labels, train_d, val_d, test_d = \
            load_data(args.dataset)
    print(features.shape)
    full_adj = torch.from_numpy(full_adj.toarray())
    print(torch.nonzero(full_adj!=full_adj.t()))
    model = gcn_sequential_model(nfeat=features.shape[1],
                                 nhid=int(128), 
                                 nclass=41,
                                 adj=full_adj)
    model.eval()
    weights={}
    keys = []
    with h5py.File("reddit_gcn_model.hdf5","r") as hf:
        hf.visit(keys.append)
        for key in keys:
            if ':' in key:
                print(hf[key].name)
                weights[hf[key].name] = hf[key].value

    hf.close()
    print(weights)
    model[0].weight.data.copy_(torch.from_numpy(weights["/model/dense0_vars/weights:0"])) 
    model[1].weight.data.copy_(torch.from_numpy(weights["/model/dense1_vars/weights:0"])) 
    features= torch.FloatTensor(features)
    #print(labels)
    labels = torch.LongTensor(labels)
    test(model, features, labels, test_d)
    #Test()

else:
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    #print(len(idx_test)) 
    #print(features.size())
    adj = adj.to_dense()
    #print(torch.nonzero(adj!=adj.t()))
    model = gcn_sequential_model(nfeat=features.shape[1],
                                 nhid=args.hidden, 
                                 nclass=labels.max().item() + 1,
                                 adj=adj)
    model_path = args.dataset+"_"+str(args.hidden)+"_gcn_model.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    preds, clean_acc = test(model, features, labels, idx_test)
    #perm =  torch.randperm(idx_test.size()[0])
    #idx = perm[1:2]
    n_neigh = args.n_neigh
    target_idx_list = select_target_node(adj, n_neigh, preds, labels, idx_test)
    #print(len(idx_test))
    #test_samples = idx_test[idx]
    #target_idx = random.choice(target_idx_list)
    #target_idx = target_idx_list[:1]
#    print("Attacking on "+str(target_idx))
    #k= 100
    #perm =  torch.randperm(idx_train.size()[0])
    #idx = perm[:k]
    #hops = args.hops
    #perturb_idx = select_perturb_node(adj, target_idx, hops, None, False)
    #perturb_idx = perturb_idx.numpy()
    #perturb_idx = select_perturb_node(adj, target_idx, hops, None, True)
    #perturb_idx = select_perturb_node(adj, target_idx, hops, 0.1, False)
    #modified_nodes_idx = idx_train[idx]
    #print("Perturbing on ")
    #print(perturb_idx)
    #mask = torch.FloatTensor(features.size())
    #mask.zero_()
    #mask[perturb_idx,:] = 1
attack = PGD(model)
epsilon = args.epsilon
#correct = 0
batch_size = args.batch_size
err_count = 0
for i in range(len(target_idx_list)//batch_size):
    if i==10:
        break
    target_idx = target_idx_list[i*batch_size:(i+1)*batch_size]
    print("Attacking on "+str(target_idx))
    hops = args.hops
    #perturb_idx = select_perturb_node(adj, target_idx, hops, None, False)
    perturb_idx = select_perturb_node(adj, target_idx, hops, None, True)
    perturb_idx = perturb_idx.numpy()
    #print("Perturbing on ")
    #print(perturb_idx)
    mask = torch.FloatTensor(features.size())
    mask.zero_()
    mask[perturb_idx,:] = 1
    xl = features.clone()
    xu = features.clone()
    xl[perturb_idx] = torch.clamp(xl[perturb_idx], min=0)
    xu[perturb_idx] = torch.clamp(xu[perturb_idx], max=1)
    bounds = GCNBoundsTwoLayer(model, features, adj, epsilon, targets=target_idx, perturb_targets=perturb_idx, elision=True, xl=xl,xu=xu)
    LB = bounds.LB
    UB = bounds.UB
    #print("error: ", elision_error(LB[-1]))
    error = elision_error(LB[-1])
    err_count += error*batch_size
    #print(err_count)
#if (len(target_idx_list)-(i+1)*batch_size)!= 0:
#    target_idx = target_idx_list[(i+1)*batch_size:]
#    print("Attacking on "+str(target_idx))
#    hops = args.hops
    #perturb_idx = select_perturb_node(adj, target_idx, hops, None, False)
#    perturb_idx = select_perturb_node(adj, target_idx, hops, None, True)
#    perturb_idx = perturb_idx.numpy()
    #print("Perturbing on ")
    #print(perturb_idx)
#    mask = torch.FloatTensor(features.size())
#    mask.zero_()
#    mask[perturb_idx,:] = 1
#    xl = features.clone()
#    xu = features.clone()
#    xl[perturb_idx] = torch.clamp(xl[perturb_idx], min=0)
#    xu[perturb_idx] = torch.clamp(xu[perturb_idx], max=1)
#    bounds = GCNBoundsTwoLayer(model, features, adj, epsilon, targets=target_idx, perturb_targets=perturb_idx, elision=True, xl=xl,xu=xu)
#    LB = bounds.LB
#    UB = bounds.UB
    #print("error: ", elision_error(LB[-1]))
#    error = elision_error(LB[-1])
#    acc_count += (1-error)*batch_size


#print(len(idx_test))
#ave_acc = acc_count / len(idx_test)
ave_err = err_count / (10*batch_size)
#print(acc_count , len(idx_test))
#print("acc_lb_bound"+str(ave_acc))
print("err_lb_bound "+str(ave_err))

