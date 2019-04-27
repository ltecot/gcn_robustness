import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid, activation=F.relu)
#         self.gc2 = GraphConvolution(nhid, nclass, activation=None)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         # x = F.relu(self.gc1(x, adj))
#         x = self.gc1(x, adj)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         # return F.log_softmax(x, dim=1)
#         return x

def gcn_sequential_model(nfeat, nhid, nclass, adj):
    model = nn.Sequential(
        GraphConvolution(nfeat, nhid, adj, activation=F.relu),
        GraphConvolution(nhid, nclass, adj, activation=None)
    )
    return model