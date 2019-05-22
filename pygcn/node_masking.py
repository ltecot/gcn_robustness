import torch
import numpy as np

def select_target_node(adj, n_neighbours, pred, label, idx_test):
    #print(idx_test)
    adj = adj[idx_test]
    mask = (pred==label[idx_test])
    #print(pred.size(0))
    adj_filter = adj[mask]
    #print(len(idx_test))
    #print(adj_filter.size(0))
    #nnz = torch.nonzero(adj_filter)
    #print(nnz.size())
    n_idx = []
    for i in range(adj_filter.size(0)):
        nnz = torch.nonzero(adj_filter[i])
        if nnz.size(0)> n_neighbours:
            n_idx.append(idx_test[mask][i].item())
    #print(n_idx)
    return sorted(n_idx)

def select_perturb_node(adj,target_node,hops, random_p, with_target_node):
    #adj = adj - torch.eye(adj.size(0))
    if random_p==None:
        row_target = adj[target_node]
        for i in range(hops):
            nnz = torch.nonzero(row_target)
            #print(nnz)
            neighbor=torch.unique(nnz[:,1])
            row_target = adj[neighbor]
        
        n_idx = torch.sort(neighbor)[0]
        if not with_target_node:
            for tn in target_node:
                n_idx = n_idx[n_idx!=tn]
    else:
        perm = torch.randperm(adj.size(0))
        k = int(adj.size(0)*random_p)
        if not with_target_node:
            for tn in target_node:
                perm = perm[perm!=tn]
        n_idx = torch.sort(perm[:k])[0]
    return n_idx
        
def select_target_node_sp(adj, n_neighbours, pred, label, idx_test, multitask=False):
    #print(idx_test)
    if multitask:
        #print(type(idx_test),idx_test)
        return sorted(idx_test)
    adj = adj[idx_test]
    #mask = (pred==label[idx_test])
    label = label[idx_test]
    #print(mask)
    #print(pred.size(0))
    #mask = mask.numpy()
    #adj_filter = adj[mask]
    
    #correct_idx = idx_test[mask]
    #print(idx_test)
    #print(len(idx_test))
    #print(adj_filter.size(0))
    #nnz = torch.nonzero(adj_filter)
    #print(nnz.size())
    n_idx = []
    for i in range(adj.shape[0]):
        if pred[i]!=label[i]:
            continue
        nnz = np.nonzero(adj[i])[1]
        #print(nnz)
        if nnz.shape[0]> n_neighbours:
            n_idx.append(idx_test[i])
    #print(n_idx)
    return sorted(n_idx)

def select_perturb_node_sp(adj,target_node,hops, random_p, with_target_node):
    #adj = adj - torch.eye(adj.size(0))
    if random_p==None:
        row_target = adj[target_node]
        for i in range(hops):
            nnz = np.nonzero(row_target)
            neighbor = nnz[1]
            #print(nnz)
            #neighbor=torch.unique(nnz[:,1])
            row_target = adj[neighbor]
        
        #n_idx = torch.sort(neighbor)[0]
        n_idx = sorted(neighbor)
        #print(n_idx)
        if not with_target_node:
            for tn in target_node:
                #print(type(tn),type(n_idx))
                n_idx = [i for i in n_idx if i!= tn]
    else:
        #perm = torch.randperm(adj.size(0))
        perm = np.random.permutation(adj.shape[0])
        k = int(adj.size(0)*random_p)
        if not with_target_node:
            for tn in target_node:
                perm = perm[perm!=tn]
        n_idx = sorted(perm[:k])
    return n_idx
        

