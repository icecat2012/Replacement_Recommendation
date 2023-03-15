import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def simple_normalize(a):
    a_norm = a.norm(2, 1, keepdim=True)
    a_norm = torch.where(a_norm == 0, torch.tensor(1.).to(a_norm), a_norm)
    a = a / a_norm
    return a

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

class Dtopk(nn.Module): #differentiable top k mask gerenation (input 2D tensor)
    def __init__(self):
        super(Dtopk, self).__init__()
        self.r = nn.ReLU(inplace=True)
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, x, num, largest=True, sort=True):
        t, t_i = torch.topk(x, num, dim=1, largest=largest, sorted=sort)
        xx = x.unsqueeze(2).repeat(1,1,t.shape[1])
        t = t.unsqueeze(1).repeat(1,x.shape[1],1)
        x_m = self.r(-(xx - t)*(xx - t) + self.eps*self.eps) / (self.eps*self.eps)

        x_m_sum = torch.sum(x_m, dim=1)
        x_m_sum = x_m_sum.unsqueeze(1).repeat(1,x_m.shape[1],1)
        x_m = self.r(x_m - x_m_sum + 1)

        return x_m

class Recommender(nn.Module):
    def __init__(self, data_channels, n_classes=400, max_group_member=5, max_scene_item=10):
        super(Recommender, self).__init__()
        # self.eps = torch.finfo(torch.float32).eps
        self.max_scene_item = max_scene_item
        self.max_group_member = max_group_member
        self.m = nn.Softmax(dim=1)
        self.merge_UU = nn.Conv2d(2,1,1,stride=1)
        self.PS = nn.Sequential(Linear_Activation(n_classes+max_group_member, data_channels, activation='leaky', bias=True), 
                                Linear_Activation(data_channels, data_channels, activation='leaky', bias=True),
                                Linear_Activation(data_channels, n_classes, activation='linear', bias=False))
        self.PH = nn.Sequential(Linear_Activation(max_group_member+max_scene_item, data_channels, activation='leaky', bias=True), 
                                Linear_Activation(data_channels, data_channels, activation='leaky', bias=True),
                                Linear_Activation(data_channels, max_group_member, activation='linear', bias=False))
        self.merge_KU = nn.Conv2d(2,1,1,stride=1)

    def forward(self, Vuser, Vitem, Kuser, Kitem, SubstituteGroupScore, Vscore, Kscore, II):
        VUU = torch.matmul(Vuser, Vuser.T) #UU
        KUU = torch.matmul(Kuser, Kuser.T) #UU
        #Vscore #UV
        #Kscore #UK
        #II #KV
        
        ItemGroups_m = self.m(SubstituteGroupScore)  #KV

        prefer = Kscore - torch.matmul(Vscore, ItemGroups_m.T) #UK
        social = self.merge_UU(torch.cat([VUU.unsqueeze(0), KUU.unsqueeze(0)], dim=0).unsqueeze(0))[0,0,:,:] #UU
        haptic = II.clone()

        prefer_social = self.PS(torch.cat([prefer, social], dim=1))
        prefer_haptic = self.PH(torch.cat([prefer.T, haptic], dim=1)).T

        prefer_social = simple_normalize(prefer_social)
        prefer_haptic = simple_normalize(prefer_haptic)

        score = self.merge_KU(torch.cat([prefer_social.unsqueeze(0), prefer_haptic.unsqueeze(0)], dim=0).unsqueeze(0))[0,0,:,:] #UK

        Substitute_m = self.m(score) #UK
        Recommended_m = self.m(torch.matmul(score, ItemGroups_m)) #UV

        return Recommended_m, Substitute_m, ItemGroups_m.T, VUU, KUU

        
#############################################
# recommend top n = 2
# latent feature = F
# (shopping user matrix) user = Uv F
# (knowledge user matrix) Kuser = Uk F
# (visual item matrix) Vitem = V F
# (knowledge item matrix) Kitem = K F

# (preference score) Vrank = V Uv
# (preference score) Krank = K Uv

# (double) plus, minus