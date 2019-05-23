import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import h5py
from utils import load_data, accuracy
from models import gcn_sequential_model
from PGD import PGD
from node_masking import select_target_node, select_perturb_node, select_target_node_sp, select_perturb_node_sp
import pickle
import os

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
parser.add_argument('--p_file', type=str, default=None,
                    help='Pickle file to be used')

parser.add_argument('--n_neigh', type=int, default=0,
                    help='number of neighbors of target node')
parser.add_argument('--start', type=int, default=0,
                    help='starting node')
parser.add_argument('--npoints', type=int, default=10,
                    help='points to be added')
parser.add_argument('--hops', type=int, default=1,
                    help='hops of neighbors of target node')
parser.add_argument('--epsilon', type=float, default=0.01,
                    help='epsilon in the PGD attack')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='verbose.')
parser.add_argument('--single', action='store_true', default=False,
                    help='single perturb')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def test_ppi(model, features, labels, idx_test):
    #model.eval()
    # output = model(features, adj)
    output = model(features)
    #print(output)
    #print(labels)
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    preds = output[idx_test] > 0
    labels = labels[idx_test] > 0.5
    correct = preds.eq(labels)
    #print(correct)
    #correct = correct.sum()
    #print(type(correct),correct)
    acc_test = torch.mean(correct.float())

    criterion = torch.nn.BCEWithLogitsLoss()
    loss_test = criterion(preds.float(), labels.float())
    #print(preds,labels)
    #loss_test = F.nll_loss(F.log_softmax(output[idx_test], dim=1), labels[idx_test])
    #acc_test = accuracy(output[idx_test], labels[idx_test])
    #acc_test =  correct/labels.shape[0] 
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
   
    #preds = output[idx_test].max(1)[1].type_as(labels[idx_test])
    #print(output[idx_test].size(), len(idx_test))
    return preds, acc_test 

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




if args.dataset == "reddit" or args.dataset== "pubmed" or args.dataset== "ppi":
    num_data, train_adj, full_adj, features, train_features, test_features, labels, train_d, val_d, test_d, adj_t = \
            load_data(args.dataset)
    if args.dataset == "reddit":
        labels = np.argmax(labels,axis=1)
    print(labels)
    print(features.shape,labels.shape)
    print(type(features))
    if args.dataset == "reddit":
        hidden = 128
        class_num = 41
    elif args.dataset == "ppi":
        hidden = 128
        class_num = 121
        multitask = True
    elif args.dataset == "pubmed":
        hidden = 32
        class_num = 3
    #class_num = int(np.max(labels))+1
    model = gcn_sequential_model(nfeat=features.shape[1],
                                 nhid= hidden, 
                                 nclass=class_num,
                                 adj=adj_t)
    model.eval()
    weights={}
    keys = []
    model_path = args.dataset + "_gcn_model.hdf5"
    with h5py.File(model_path,"r") as hf:
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
    print(torch.max(features),torch.min(features))
    labels = torch.LongTensor(labels)
    if args.dataset == "ppi":
        preds, clean_acc = test_ppi(model, features, labels, test_d)
    else:
        preds, clean_acc = test(model, features, labels, test_d)
    #Test()
    adj = adj_t
    adj = adj.to_dense()

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
 


starting_point = args.start
npoints = args.npoints
end_point = starting_point + npoints
point_list = pickle.load(open(args.p_file,"rb"))
np.random.shuffle(point_list)
target_list = point_list[starting_point:end_point]
attack = PGD(model)
epsilon = args.epsilon
#correct = 0
dir_name = "logs/"+args.dataset+"_"+str(epsilon)+"_"+str(args.hops)+"_"+str(args.single)
os.makedirs(dir_name, exist_ok=True)
filename = dir_name+"/"+str(starting_point)+".log"
log_f = open(filename,"w")

if args.dataset == "ppi":
    multitask = True
else:
    multitask = False
for target_idx in target_list:
    print("Attacking on "+str(target_idx))
    target_idx = [target_idx]
    hops = args.hops
    if not args.single:
        perturb_idx = select_perturb_node(adj, target_idx, hops, None, False)
        perm =  torch.randperm(perturb_idx.size(0))
        perturb_idx = perturb_idx[perm[:20]]
    else:
        perturb_idx = torch.LongTensor(target_idx)
    #perturb_idx = torch.LongTensor(target_idx)
        #perturb_idx = torch.LongTensor(target_idx)
    print("Perturbing on "+ str(perturb_idx.size(0)/features.size(0)*100)+" nodes")
    perturb_idx = perturb_idx.numpy()
    #print(perturb_idx)
    mask = torch.FloatTensor(features.size())
    mask.zero_()
    mask[perturb_idx,:] = 1
    if args.dataset=="reddit":
        features_adv = attack(features, labels, target_idx, mask, epsilon, sparse=True)
    else:
        features_adv = attack(features, labels, target_idx, mask, epsilon, sparse=False, multitask=multitask)
    _, acc= test(model, features_adv, labels, target_idx)
    if acc==0:
        log_f.write("success!\n")
    else:
        log_f.write("fail!\n")

log_f.close()
#print('job start, attack point {} to {}'.format(starting_point, end_point))
#t = 1 + random.randint(0,10)
#print('job starting at point {} will run {} seconds'.format(starting_point, t))
#time.sleep(t)
#print('job starting at point {} done'.format(starting_point))
