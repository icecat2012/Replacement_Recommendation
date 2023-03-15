
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.utils import dense_to_sparse

from torch_geometric.nn import GCNConv, GATConv
from .SAGPool import SAGPool, GNNBlock, Generate_dynamic_A
from .prediction import Recommender
from .GAT import GATv2Conv
from .transformer import Transformer, Connection
from .SubstituteGroup import SubstituteGroup

def simple_normalize(a):
    a_norm = a.norm(2, 1, keepdim=True)
    a_norm = torch.where(a_norm == 0, torch.tensor(1.).to(a_norm), a_norm)
    a = a / a_norm
    return a

class Yolo_GCN(nn.Module):
    def __init__(self, n_classes=400, inference=False, n_users=None, n_items=None, max_group_member=5, max_scene_item=10):
        super(Yolo_GCN, self).__init__()

        self.n_classes = n_classes

        NodeFeatures = 256

        self.n_users = n_users
        self.n_items = n_items
        self.max_group_member = max_group_member

        self.GNN1_1 = GATv2Conv(NodeFeatures, NodeFeatures, edge_dim=1, debug=0, bias=True)
        self.GNN1_2 = GATv2Conv(NodeFeatures, NodeFeatures, edge_dim=1, debug=0, bias=True)
        self.GNN2_1 = GATv2Conv(NodeFeatures, NodeFeatures, edge_dim=1, debug=0, bias=True)
        self.GNN2_2 = GATv2Conv(NodeFeatures, NodeFeatures, edge_dim=1, debug=0, bias=True)
        self.align = Connection(NodeFeatures, N=2, heads=1, dropout=0, pe_mode=1)
        self.r = nn.LeakyReLU(0.2)

    def forward(self, Graph1, Graph2, Vitem, Kitem):
        Graph1, Edge_index1, batch1, n_id1, Edge_attr1 = Graph1.x, Graph1.edge, Graph1.batch, Graph1.nid, Graph1.value
        Graph2, Edge_index2, batch2, n_id2, Edge_attr2 = Graph2.x, Graph2.edge, Graph2.batch, Graph2.nid, Graph2.value
        
        Vgraph = self.GNN1_1(Graph1, Edge_index1, Edge_attr1)
        Kgraph = self.GNN2_1(Graph2, Edge_index2, Edge_attr2)
        
        Vgraph = simple_normalize(Vgraph)
        Kgraph = simple_normalize(Kgraph)

        Vnew = torch.zeros_like(Kgraph)
        Vnew[:self.max_group_member,:] = Vgraph[:self.max_group_member,:]
        pn = Kgraph.shape[0]-self.n_classes
        Vitem = Vitem+pn
        Kitem = Kitem+pn

        Vnew[Vitem] = Vgraph[self.max_group_member:]
        mask = [i for i in range(self.max_group_member)]
        mask.extend(Vitem)
        
        Vgraph, Kgraph = self.align(torch.unsqueeze(Vnew, 0), torch.unsqueeze(Kgraph, 0), norm=True, dropout=False, encoder_mask=mask)
        
        Vgraph = Vgraph[0]
        Kgraph = Kgraph[0]

        Vgraph = self.GNN1_2(Vgraph, Edge_index1, Edge_attr1)
        Kgraph = self.GNN2_2(Kgraph, Edge_index2, Edge_attr2)

        Vgraph = simple_normalize(Vgraph)
        Kgraph = simple_normalize(Kgraph)

        Vuser = Vgraph[:self.max_group_member, :]
        Vitem = Vgraph[self.max_group_member:, :]
        Kuser = Kgraph[:-self.n_classes, :]
        Kitem = Kgraph[-self.n_classes:,:]
        return Vuser, Vitem, Kuser, Kitem

class Group_Recommendation(nn.Module):
    def __init__(self, n_classes=400, max_group_member=5, max_scene_item=10):
        super(Group_Recommendation, self).__init__()
        NodeFeatures = 256
        self.n_classes = n_classes
        self.max_scene_item = max_scene_item
        self.ItemGroup = SubstituteGroup(NodeFeatures, max_group_member, max_scene_item, n_classes)
        self.Predict = Recommender(NodeFeatures, n_classes=n_classes, max_group_member=max_group_member, max_scene_item=max_scene_item)

    def forward(self, Vuser, Vitem, Kuser, Kitem):
        Vscore = torch.matmul(Vuser, Vitem.T)  #UV
        Kscore = torch.matmul(Kuser, Kitem.T)  #UK
        II = torch.matmul(Kitem, Vitem.T) #KV
        if self.max_scene_item>Vitem.shape[0]:
            zeros = torch.zeros(Kitem.shape[0], self.max_scene_item-Vitem.shape[0])
            if II.is_cuda:
                zeros = zeros.to(II.device)
            II = torch.cat([II, zeros], dim=1) #KV
            
        SubstituteGroupScore = self.ItemGroup(Vitem, Kitem, Kuser, Vscore, Kscore, II)
        Recommended_m, Substitute_m, ItemGroups_m, VUU, KUU = self.Predict(Vuser, Vitem, Kuser, Kitem, SubstituteGroupScore, Vscore.clone(), Kscore.clone(), II.clone())

        return Recommended_m, Substitute_m, ItemGroups_m, VUU, KUU, Vscore, Kscore, II
