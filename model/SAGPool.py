from torch_geometric.nn import GCNConv, GATConv
from .GAT import GATv2Conv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter
import torch
from torch import nn
from torch.nn import functional as F

class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GATv2Conv,non_linearity=torch.tanh, users=5):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1,edge_dim=1)
        self.non_linearity = non_linearity
        self.users = int(users)
        self.user_id = torch.Tensor([i for i in range(users)]).type(torch.long)

    def forward(self, x, edge_index, edge_attr=None, batch=None, n_id=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
            

        score = self.score_layer(x,edge_index).squeeze()
        perm = topk(score[self.users:], self.ratio, batch[self.users:])
        perm = torch.cat([perm, self.user_id], dim=0)
        x = x[perm,:] * self.non_linearity(score[perm]).view(-1, 1)
        n_id = n_id[perm]

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, n_id

class GNNBlock(nn.Module):
    def __init__(self, users, NodeFeatures, LatenFeatures, ratio=0.9, Conv=GATv2Conv):
        super(GNNBlock, self).__init__()
        self.GNN_stream1 = Conv(NodeFeatures, LatenFeatures, edge_dim=1)
        self.GNN_stream2 = SAGPool(LatenFeatures, ratio=ratio, Conv=Conv, users=users)
        self.r = nn.ReLU(inplace=True)
    def forward(self, Graph, Edge_index, edge_attr, batch, n_id):
        graph = self.r(self.GNN_stream1(Graph, Edge_index, edge_attr))
        graph, Edge_index, edge_attr, batch, perm, n_id = self.GNN_stream2(graph, Edge_index, edge_attr, batch, n_id)
        return graph, Edge_index, edge_attr, batch, perm, n_id

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class Generate_dynamic_A(nn.Module):
    def __init__(self, Node, F_size):
        super(Generate_dynamic_A, self).__init__()
        self.norm1 = Norm(F_size)
        self.norm2 = Norm(F_size)
        self.conv = nn.Conv1d(Node, Node, 1)
    def forward(self, x):
        x1 = self.norm1(x)
        x2 = self.norm2(x)
        x3 = torch.bmm(x1, x2.transpose(1,2))
        return F.relu(self.conv(x3))


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

if __name__=='__main__':
    from torch_geometric.datasets import TUDataset
    from torch_geometric.data import DataLoader
    from torch_geometric import utils
    from torch.utils.data import random_split
    import os

    dataset = TUDataset(os.path.join('./','DD'),name='DD')
    num_training = int(len(dataset)*0.8)
    num_val = int(len(dataset)*0.1)
    num_test = len(dataset) - (num_training+num_val)
    training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])

    train_loader = DataLoader(training_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(validation_set,batch_size=128,shuffle=False)
    test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
    model = SAGPool(dataset.num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)


    model.eval()
    for data in test_loader:
        x, edge_index, edge_attr, perm = model(data.x, data.edge_index, None)
        print(data.x.shape)
        print(data.edge_index.shape)
        print(data.batch.shape)
        print('################################')
        print(x.shape)
        print(edge_index.shape)
        #print(edge_attr.shape)
        print(perm.shape)
        break