import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
import pickle

# Util to compare matricies.

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.autograd import Variable

# import os.path
# import argparse

# import numpy as np

# with open("mnist_conv_adv_matrices.pkl", "wb") as f:
#                 pickle.dump({'A_LB': -matrix, 'b_LB': bias}, f)

# if __name__ == "__main__": 
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--pickle1', default="../examples/mnist_conv_adv_matrices.pkl")
#     # parser.add_argument('--pickle2', default="../../VerificationExp/fastlin_last_matrices.pkl")
#     parser.add_argument('--pickle2', default="../../VerificationExp/crown_last_matrices.pkl")
#     args = parser.parse_args()

#     if args.pickle1 == "" or args.pickle2 == "":
#         raise ValueError("Need to imput two pickle files.")

#     pickle1 = pickle.load(open(args.pickle1, "rb"))
#     pickle2 = pickle.load(open(args.pickle2, "rb"))
def compare_matricies(pickle1, pickle2):
    for k in pickle1.keys() & pickle2.keys():
        print("\n Comparing", k)
        k1 = torch.tensor(pickle1[k])
        # print(pickle2[k])
        k2 = torch.tensor(pickle2[k])
        # print(pickle1[k].shape)
        # print(pickle2[k].shape)
        print("Difference Sum: ", torch.sum(k1 - k2))
        print("Abs Difference Sum: ", torch.sum(torch.abs(k1 - k2)))
        print("Avg Sum: ", torch.sum(torch.abs(k1 - k2)) / k1.view(-1).shape[0])
        # print("Max Difference: ", torch.max(torch.abs(k1 - k2), 0))
        print("Max Difference: ", torch.max(torch.abs(k1 - k2)))

def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def generate_adj(file_name):
    fr = open(file_name,"r")
    fw = open(file_name+"_processed","w")
    idx = -100000
    idx_dict = {}
    for line in fr:
        items = line.split()
        new_item = []
        for item in items:
            if not item.isdigit():
                if item not in idx_dict:
                    idx_dict[item]=str(idx)
                    new_item.append(str(idx))
                    idx+=1
                else:
                    new_item.append(idx_dict[item])
            else:
                new_item.append(str(item))
        output = " ".join(new_item)+"\n"
        fw.write(output)
    fr.close()
    fw.close()
    return idx_dict

def encode_onehot(labels, idx_dict=None):
    if idx_dict is not None:
        labels = idx_dict[labels]
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def data_loader(dataset):
    path = "../data/"+dataset+"/"
    if dataset=="cora":
        print('Loading {} dataset...'.format(dataset))
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
    if dataset=="WebKB" or dataset=="citeseer":
        if dataset=="WebKB":
            datasets = ["cornell","texas","washington","wisconsin"]
            dataset = datasets[0]
        print('Loading {} dataset...'.format(dataset))
        # mapping non-digit nodes id to a negative number
        idx_dict = generate_adj(path+dataset+".cites")
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        # build graph
        # remapping non-digit nodes id in content
        for i in range(idx_features_labels.shape[0]):
            if idx_features_labels[i,0].isdigit():
                continue
            if idx_features_labels[i,0] not in idx_dict:
                print(idx_features_labels[i,0])
                idx_dict[i]=np.zeros(idx_features_labels.shape[1])
            else:
                idx_features_labels[i,0] = idx_dict[idx_features_labels[i,0]]

        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        #idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
        idx_map = {j: i for i, j in enumerate(idx)}
        #idx_map = {int(j): i for i, j in enumerate(list(idx_dict.values()))}
        #edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
        #                                dtype=np.int32)
        edges_unordered = np.genfromtxt("{}{}.cites_processed".format(path, dataset),
                                         dtype=np.int32)
        # delete nodes without features
        idx_list = list(map(idx_map.get, edges_unordered.flatten()))
        for i, idx in enumerate(idx_list):
            if idx is None:
                if not i%2==0:
                    idx_list[i-1]=None
                else:
                    idx_list[i+1]=None
        idx_list = [idx for idx in idx_list if idx is not None]
        edges = np.array(idx_list,dtype=np.int32).reshape(-1,2)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
 
    if dataset=="polblogs":
        file_name = "../data/"+dataset+"/"+dataset+".npz"
        with np.load(file_name) as loader:
            loader = dict(loader)
            adj  = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                                  loader['adj_indptr']), shape=loader['adj_shape'])

            if 'attr_data' in loader:
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                       loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                print("no feature")
                features = None
            labels = loader.get('labels')

    return adj, features, labels

def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None, random_state=42):

    """
    Split the arrays or matrices into random train, validation and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels.
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;

    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    return idx_train, idx_val, idx_test

def load_data(dataset="cora"):
    
    adj, features, labels = data_loader(dataset)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #print(features)
    #features = normalize(features)
    #print(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train, idx_val, idx_test = train_val_test_split_tabular(np.arange(adj.shape[0]))
    #print(idx_train,idx_val,idx_test)
    #idx_train = range(140)
    #idx_val = range(200, 500)
    #idx_test = range(500, 1500)

   
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    

        
    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    #print(preds,labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
