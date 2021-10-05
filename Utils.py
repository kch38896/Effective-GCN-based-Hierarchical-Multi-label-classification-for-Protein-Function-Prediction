import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Embedder(nn.Module):
    def __init__(self, go_size, hidden_dimension):
        super().__init__()
        self.embed = nn.Linear(go_size,hidden_dimension)

    def forward(self, x):
        node_feature = self.embed(x)
        node_feature = F.normalize(node_feature)
        return node_feature


class FC(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.Linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        seq_feature = self.Linear(x)

        return seq_feature

class GraphConvolution(nn.Module):
    def __init__(self, nfeat, nhid, bias=True):
        super(GraphConvolution, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.weight = Parameter(torch.FloatTensor(nfeat, nhid))
        if bias:
            self.bias = Parameter(torch.FloatTensor(nhid))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = torch.mm(input, self.weight)
        output = torch.spmm(adj, x)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        return x
