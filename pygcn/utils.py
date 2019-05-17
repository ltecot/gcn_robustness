import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
import os
from time import time
import pickle

# Computes error associated with elision bound output.
# Takes in the lower derived bounds.
def elision_error(LB):
    n_ep = -1e-10  # Small neg non-zero for numerical error
    N = LB.shape[0]
    err = 0
    for n in range(N):
        if torch.sum(LB[n] < n_ep) > 0:
            err += 1
    return err / N

def compare_matricies(pickle1, pickle2):
    for k in pickle1.keys() & pickle2.keys():
        print("\n Comparing", k)
        k1 = torch.tensor(pickle1[k])
        # print(pickle2[k])
        k2 = torch.tensor(pickle2[k])
        # print(k1.shape)
        # print(k2.shape)
        print("Difference Sum: ", torch.sum(k1 - k2))
        print("Abs Difference Sum: ", torch.sum(torch.abs(k1 - k2)))
        print("Avg Difference Sum: ", torch.sum(torch.abs(k1 - k2)) / k1.view(-1).shape[0])
        # print("Max Difference: ", torch.max(torch.abs(k1 - k2), 0))
        print("Max Difference: ", torch.max(torch.abs(k1 - k2)))

# Only takes 1D vectors
def tensor_product(A, B):
    return torch.einsum("a,b->ab", A, B).view(-1)

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
    classes = sorted(set(labels))
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
    if dataset == "reddit":
        prefix = "../data/reddit/reddit"
        return load_graphsage_data(prefix)
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


def load_gcn_data(dataset_str):
    npz_file = 'data/{}/{}_{}.npz'.format(dataset_str, dataset_str, "gcn")
    if os.path.exists(npz_file):
        start_time = time()
        print('Found preprocessed dataset {}, loading...'.format(npz_file))
        data = np.load(npz_file)
        num_data     = data['num_data']
        labels       = data['labels']
        train_data   = data['train_data']
        val_data     = data['val_data']
        test_data    = data['test_data']
        train_adj = sp.csr_matrix((data['train_adj_data'], data['train_adj_indices'], data['train_adj_indptr']), shape=data['train_adj_shape'])
        full_adj = sp.csr_matrix((data['full_adj_data'], data['full_adj_indices'], data['full_adj_indptr']), shape=data['full_adj_shape'])
        feats = sp.csr_matrix((data['feats_data'], data['feats_indices'], data['feats_indptr']), shape=data['feats_shape'])
        train_feats = sp.csr_matrix((data['train_feats_data'], data['train_feats_indices'], data['train_feats_indptr']), shape=data['train_feats_shape'])
        test_feats = sp.csr_matrix((data['test_feats_data'], data['test_feats_indices'], data['test_feats_indptr']), shape=data['test_feats_shape'])
        print('Finished in {} seconds.'.format(time() - start_time))
    else:
        """Load data."""
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        if dataset_str != 'nell':
            test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
            test_idx_range = np.sort(test_idx_reorder)

            if dataset_str == 'citeseer':
                # Fix citeseer dataset (there are some isolated nodes in the graph)
                # Find isolated nodes, add them as zero-vecs into the right position
                test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
                tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
                tx_extended[test_idx_range-min(test_idx_range), :] = tx
                tx = tx_extended
                ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
                ty_extended[test_idx_range-min(test_idx_range), :] = ty
                ty = ty_extended

            features = sp.vstack((allx, tx)).tolil()
            features[test_idx_reorder, :] = features[test_idx_range, :]
            adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

            labels = np.vstack((ally, ty))
            labels[test_idx_reorder, :] = labels[test_idx_range, :]

            idx_test = test_idx_range.tolist()
            idx_train = range(len(y))
            idx_val = range(len(y), len(y)+500)

            train_mask = sample_mask(idx_train, labels.shape[0])
            val_mask = sample_mask(idx_val, labels.shape[0])
            test_mask = sample_mask(idx_test, labels.shape[0])

            y_train = np.zeros(labels.shape)
            y_val = np.zeros(labels.shape)
            y_test = np.zeros(labels.shape)

            y_test[test_mask, :] = labels[test_mask, :]
        else:
            test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
            features = allx.tocsr()
            adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
            labels = ally
            idx_test = test_idx_reorder
            idx_train = range(len(y))
            idx_val = range(len(y), len(y)+969)
            train_mask = sample_mask(idx_train, labels.shape[0])
            val_mask = sample_mask(idx_val, labels.shape[0])
            test_mask = sample_mask(idx_test, labels.shape[0])
            y_train = np.zeros(labels.shape)
            y_val = np.zeros(labels.shape)
            y_test = np.zeros(labels.shape)
            y_train[train_mask, :] = labels[train_mask, :]
            y_val[val_mask, :] = labels[val_mask, :]
            y_test[test_mask, :] = labels[test_mask, :]

        # num_data, (v, coords), feats, labels, train_d, val_d, test_d
        num_data = features.shape[0]
        def _normalize_adj(adj):
            rowsum = np.array(adj.sum(1)).flatten()
            d_inv  = 1.0 / (rowsum+1e-20)
            d_mat_inv = sp.diags(d_inv, 0)
            adj = d_mat_inv.dot(adj).tocoo()
            coords = np.array((adj.row, adj.col)).astype(np.int32)
            return adj.data.astype(np.float32), coords

        def gcn_normalize_adj(adj):
            adj = adj + sp.eye(adj.shape[0])
            rowsum = np.array(adj.sum(1)) + 1e-20
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
            adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            adj = adj.tocoo()
            coords = np.array((adj.row, adj.col)).astype(np.int32)
            return adj.data.astype(np.float32), coords

        # Normalize features
        rowsum = np.array(features.sum(1)) + 1e-9
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv, 0)
        features = r_mat_inv.dot(features)

        #if FLAGS.normalization == 'gcn':
        full_v, full_coords = gcn_normalize_adj(adj)
        full_v = full_v.astype(np.float32)
        full_coords = full_coords.astype(np.int32)
        train_v, train_coords = full_v, full_coords
        labels = (y_train + y_val + y_test).astype(np.float32)
        train_data = np.nonzero(train_mask)[0].astype(np.int32)
        val_data   = np.nonzero(val_mask)[0].astype(np.int32)
        test_data  = np.nonzero(test_mask)[0].astype(np.int32)

        feats = (features.data, features.indices, features.indptr, features.shape)

        def _get_adj(data, coords):
            adj = sp.csr_matrix((data, (coords[0,:], coords[1,:])), 
                                shape=(num_data, num_data))
            return adj

        train_adj = _get_adj(train_v, train_coords)
        full_adj  = _get_adj(full_v,  full_coords)
        feats = sp.csr_matrix((feats[0], feats[1], feats[2]), 
                              shape=feats[-1], dtype=np.float32)

        train_feats = train_adj.dot(feats)
        test_feats  = full_adj.dot(feats)

        with open(npz_file, 'wb') as fwrite:
            np.savez(fwrite, num_data=num_data, 
                             train_adj_data=train_adj.data, train_adj_indices=train_adj.indices, train_adj_indptr=train_adj.indptr, train_adj_shape=train_adj.shape,
                             full_adj_data=full_adj.data, full_adj_indices=full_adj.indices, full_adj_indptr=full_adj.indptr, full_adj_shape=full_adj.shape,
                             feats_data=feats.data, feats_indices=feats.indices, feats_indptr=feats.indptr, feats_shape=feats.shape,
                             train_feats_data=train_feats.data, train_feats_indices=train_feats.indices, train_feats_indptr=train_feats.indptr, train_feats_shape=train_feats.shape,
                             test_feats_data=test_feats.data, test_feats_indices=test_feats.indices, test_feats_indptr=test_feats.indptr, test_feats_shape=test_feats.shape,
                             labels=labels,
                             train_data=train_data, val_data=val_data, 
                             test_data=test_data)

    return num_data, train_adj, full_adj, feats, train_feats, test_feats, labels, train_data, val_data, test_data


def load_graphsage_data(prefix, normalize=True):
    version_info = list(map(int, nx.__version__.split('.')))
    major = version_info[0]
    minor = version_info[1]
    assert (major <= 1) and (minor <= 11), "networkx major version must be <= 1.11 in order to load graphsage data"

    # Save normalized version
    max_degree = -1
    if max_degree==-1:
        npz_file = prefix + '.npz'
    else:
        npz_file = '{}_deg{}.npz'.format(prefix, max_degree)

    if os.path.exists(npz_file):
        start_time = time()
        print('Found preprocessed dataset {}, loading...'.format(npz_file))
        data = np.load(npz_file)
        num_data     = data['num_data']
        feats        = data['feats']
        train_feats  = data['train_feats']
        test_feats   = data['test_feats']
        labels       = data['labels']
        train_data   = data['train_data']
        val_data     = data['val_data']
        test_data    = data['test_data']
        train_adj = sp.csr_matrix((data['train_adj_data'], data['train_adj_indices'], data['train_adj_indptr']), shape=data['train_adj_shape'])
        full_adj  = sp.csr_matrix((data['full_adj_data'], data['full_adj_indices'], data['full_adj_indptr']), shape=data['full_adj_shape'])
        print('Finished in {} seconds.'.format(time() - start_time))
    else:
        print('Loading data...')
        start_time = time()
    
        G_data = json.load(open(prefix + "-G.json"))
        G = json_graph.node_link_graph(G_data)
    
        feats = np.load(prefix + "-feats.npy").astype(np.float32)
        id_map = json.load(open(prefix + "-id_map.json"))
        if id_map.keys()[0].isdigit():
            conversion = lambda n: int(n)
        else:
            conversion = lambda n: n
        id_map = {conversion(k):int(v) for k,v in id_map.iteritems()}

        walks = []
        class_map = json.load(open(prefix + "-class_map.json"))
        if isinstance(class_map.values()[0], list):
            lab_conversion = lambda n : n
        else:
            lab_conversion = lambda n : int(n)
    
        class_map = {conversion(k): lab_conversion(v) for k,v in class_map.iteritems()}

        ## Remove all nodes that do not have val/test annotations
        ## (necessary because of networkx weirdness with the Reddit data)
        broken_count = 0
        to_remove = []
        for node in G.nodes():
            if not id_map.has_key(node):
            #if not G.node[node].has_key('val') or not G.node[node].has_key('test'):
                to_remove.append(node)
                broken_count += 1
        for node in to_remove:
            G.remove_node(node)
        print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))
    
        # Construct adjacency matrix
        print("Loaded data ({} seconds).. now preprocessing..".format(time()-start_time))
        start_time = time()
    
        edges = []
        for edge in G.edges():
            if id_map.has_key(edge[0]) and id_map.has_key(edge[1]):
                edges.append((id_map[edge[0]], id_map[edge[1]]))
        print('{} edges'.format(len(edges)))
        num_data   = len(id_map)

        if max_degree != -1:
            print('Subsampling edges...')
            edges = subsample_edges(edges, num_data, FLAGS.max_degree)

        val_data   = np.array([id_map[n] for n in G.nodes() 
                                 if G.node[n]['val']], dtype=np.int32)
        test_data  = np.array([id_map[n] for n in G.nodes() 
                                 if G.node[n]['test']], dtype=np.int32)
        is_train   = np.ones((num_data), dtype=np.bool)
        is_train[val_data] = False
        is_train[test_data] = False
        train_data = np.array([n for n in range(num_data) if is_train[n]], dtype=np.int32)
        
        train_edges = [(e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]]
        edges       = np.array(edges, dtype=np.int32)
        train_edges = np.array(train_edges, dtype=np.int32)
    
        # Process labels
        if isinstance(class_map.values()[0], list):
            num_classes = len(class_map.values()[0])
            labels = np.zeros((num_data, num_classes), dtype=np.float32)
            for k in class_map.keys():
                labels[id_map[k], :] = np.array(class_map[k])
        else:
            num_classes = len(set(class_map.values()))
            labels = np.zeros((num_data, num_classes), dtype=np.float32)
            for k in class_map.keys():
                labels[id_map[k], class_map[k]] = 1
    
        if normalize:
            from sklearn.preprocessing import StandardScaler
            train_ids = np.array([id_map[n] for n in G.nodes() 
                          if not G.node[n]['val'] and not G.node[n]['test']])
            train_feats = feats[train_ids]
            scaler = StandardScaler()
            scaler.fit(train_feats)
            feats = scaler.transform(feats)

        def _normalize_adj(edges):
            adj = sp.csr_matrix((np.ones((edges.shape[0]), dtype=np.float32),
                (edges[:,0], edges[:,1])), shape=(num_data, num_data))
            adj += adj.transpose()

            rowsum = np.array(adj.sum(1)).flatten()
            d_inv  = 1.0 / (rowsum+1e-20)
            d_mat_inv = sp.diags(d_inv, 0)
            adj = d_mat_inv.dot(adj).tocoo()
            coords = np.array((adj.row, adj.col)).astype(np.int32)
            return adj.data, coords

        train_v, train_coords = _normalize_adj(train_edges)
        full_v,  full_coords  = _normalize_adj(edges)

        def _get_adj(data, coords):
            adj = sp.csr_matrix((data, (coords[0,:], coords[1,:])),
                                shape=(num_data, num_data))
            return adj
        
        train_adj = _get_adj(train_v, train_coords)
        full_adj  = _get_adj(full_v,  full_coords)
        train_feats = train_adj.dot(feats)
        test_feats  = full_adj.dot(feats)

        print("Done. {} seconds.".format(time()-start_time))
        with open(npz_file, 'wb') as fwrite:
            #print('Saving {} edges'.format(full_adj.nnz))
            np.savez(fwrite, num_data=num_data, 
                             train_adj_data=train_adj.data, train_adj_indices=train_adj.indices, train_adj_indptr=train_adj.indptr, train_adj_shape=train_adj.shape,
                             full_adj_data=full_adj.data, full_adj_indices=full_adj.indices, full_adj_indptr=full_adj.indptr, full_adj_shape=full_adj.shape,
                             feats=feats, train_feats=train_feats, test_feats=test_feats,
                             labels=labels,
                             train_data=train_data, val_data=val_data, 
                             test_data=test_data)

    return num_data, train_adj, full_adj, feats, train_feats, test_feats, labels, train_data, val_data, test_data







def load_data(dataset="cora"):
   
    if dataset == "reddit":
        return data_loader(dataset)
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
