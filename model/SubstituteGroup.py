import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from .transformer import Norm

class Linear_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, activation, bias=False):
        super().__init__()

        self.layers = list()
        self.layers.append(nn.Linear(in_channels, out_channels, bias=bias))

        if activation == "relu":
            self.layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.layers.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            'Do nothing'

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

def QKVattention(q, k, v, d_k):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    output = torch.matmul(scores, v)
    return output

class block(nn.Module):
    def __init__(self, d_k, dropout=0.1, d_l=512): #d_l: latent feature size, d_k: input feature size
        super(block, self).__init__()
        self.feedforward = nn.Sequential(nn.Linear(d_k, d_l),nn.ReLU(inplace=True),nn.Dropout(dropout),nn.Linear(d_l, d_k))
        self.norm_1 = Norm(d_k)
        self.norm_2 = Norm(d_k)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, q, k, v, d_k, norm=True, dropout=True):
        if norm and dropout:
            q2 = self.norm_1(q)
            k2 = self.norm_1(k)
            v2 = self.norm_1(v)
            q = q + self.dropout_1(QKVattention(q2,k2,v2, d_k))
            q2 = self.norm_2(q)
            q = q + self.dropout_2(self.feedforward(q2))
        elif norm:
            q2 = self.norm_1(q)
            k2 = self.norm_1(k)
            v2 = self.norm_1(v)
            q = q + QKVattention(q2,k2,v2, d_k)
            q2 = self.norm_2(q)
            q = q + self.feedforward(q2)
        elif dropout:
            q = q + self.dropout_1(QKVattention(q,k,v, d_k))
            q = q + self.dropout_2(self.feedforward(q))
        else:
            q = q + QKVattention(q,k,v, d_k)
            q = q + self.feedforward(q)
        return q

class SubstituteGroup(nn.Module):
    def __init__(self, item_channels, max_group_member=5, max_scene_item=10, substitute_num=400):
        super(SubstituteGroup, self).__init__()
        self.item_channels = item_channels
        self.project1 = nn.Sequential(Linear_Activation(item_channels, item_channels, activation='leaky', bias=True), Linear_Activation(item_channels, item_channels, activation='linear', bias=False))
        self.project2 = nn.Sequential(Linear_Activation(item_channels, item_channels, activation='leaky', bias=True), Linear_Activation(item_channels, item_channels, activation='linear', bias=False))
        self.project3 = nn.Sequential(Linear_Activation(item_channels+max_group_member+max_scene_item, item_channels, activation='leaky', bias=True), 
                                        Linear_Activation(item_channels, item_channels, activation='leaky', bias=True), 
                                        Linear_Activation(item_channels, item_channels, activation='linear', bias=False))

        self.CNN = nn.Conv2d(max_group_member,1,1,1)
        self.block1 = block(item_channels)
        self.block2 = block(item_channels)
        self.block3 = block(item_channels)
        self.norm_1 = Norm(item_channels)

        self.r = nn.ReLU(inplace=True)
        self.linear = nn.Linear(item_channels, substitute_num)

    def forward(self, Vitem, Kitem, Kuser, Vscore, Kscore, II):
        structure = II.clone()
        preference = Kscore.clone().T
        Vitem1 = self.project1(Vitem)
        Kitem1 = self.project2(Kitem)

        ItemGroups = torch.zeros(Kuser.shape[0], Kitem.shape[0], Vitem.shape[0])
        if Vitem.is_cuda:
            ItemGroups = ItemGroups.to(Vitem.device)
        for i in range(Kuser.shape[0]):
            key = self.project3(torch.cat([(Kuser[i:i+1,:]).repeat(structure.shape[0],1), preference, structure],dim=1))
            ItemGroup = self.block1(Vitem1, key, Kitem1, self.item_channels, norm=True, dropout=False)
            ItemGroup = self.block2(ItemGroup, ItemGroup, ItemGroup, self.item_channels, norm=True, dropout=False)
            ItemGroup = self.block3(ItemGroup, ItemGroup, ItemGroup, self.item_channels, norm=True, dropout=False)
            ItemGroup = self.linear(self.r(ItemGroup))
            # ItemGroup = Norm(ItemGroup)
            ItemGroups[i,:,:] += ItemGroup.T

        return self.CNN(ItemGroups.unsqueeze(0))[0,0,:,:]


