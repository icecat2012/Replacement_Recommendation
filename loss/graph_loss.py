#https://zhuanlan.zhihu.com/p/148262580
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def BuildInvIndex(index):
    d = dict()
    cnt= 0 
    for i in index:
        d[i.item()] = cnt
        cnt+=1
    return d

def ranking_loss(score_predict,score_real):
    """
    Calculate the loss of ranknet without weight
    :param score_predict: 1xN tensor with model output score
    :param score_real: 1xN tensor with real score
    :return: Loss of ranknet
    """
    score_diff = torch.sigmoid(score_predict - score_predict.t())
    tij = (1.0 + torch.sign(score_real - score_real.t())) / 2.0
    loss_mat = tij * torch.log(score_diff) + (1-tij)*torch.log(1-score_diff)
    return -loss_mat.sum()

def TruePreference(e, v, Recommended_m, Replaced_m, Substitute_m, invVuser, invKuser, invVitem, invKitem):
    Recommend_i = torch.argmax(Recommended_m, dim=1)
    Replace_i = torch.argmax(Replaced_m, dim=1)
    Substitute_i = torch.argmax(Substitute_m, dim=1)
    Substitute_max,_ = torch.max(Substitute_m, dim=1)

    Preference_s = torch.ones_like(Recommend_i) * torch.finfo(torch.float32).eps

    Recommend = torch.zeros_like(Recommend_i)
    Replace = torch.zeros_like(Replace_i)
    Substitute = torch.zeros_like(Substitute_i)

    for j in range(e.shape[1]):
        u = e[0,j]
        i = e[1,j]
        if u in invVuser:
            u = invVuser[u]
            Preference_s[u, 0] += v[j]
            if i in invVitem:
                i = invVitem[i]
                for k in range(Recommend_i.shape[1]):
                    if i==Recommend_i[u,k]:
                        Recommend[u,k] = v[j]
                        break
                for k in range(Replace_i.shape[1]):
                    if i==Replace_i[u,k]:
                        Replace[u,k] = v[j]
                        break
            elif i in invKitem:
                i = invKitem[i]
                for k in range(Substitute_i.shape[1]):
                    if Substitute_max[u,k]>0.001 and i==Substitute_i[u,k]:
                        Substitute[u,k] = v[j]
                        break
    for i in range(1,Preference_s.shape[1],1):
        Preference_s[:,i] = Preference_s[:,0]

    Recommend = Recommend / Preference_s
    Replace = Replace / Preference_s
    Substitute = Substitute / Preference_s
    return Recommend, Replace, Substitute

def contrast_loss(padj, nadj, px, nx):
    pcnt = padj.clone()
    pcnt[pcnt>0]=1
    pcnt[pcnt<0]=0
    pcnt = torch.sum(pcnt, dim=1).int()
    ncnt = nadj.clone()
    ncnt[ncnt>0]=1
    ncnt[ncnt<0]=0
    ncnt = torch.sum(ncnt, dim=1).int()
    pval, pinc = torch.sort(padj, dim=1)
    # ncnt = torch.sum(nadj, dim=1)
    nval, ninc = torch.sort(nadj, dim=1)
    loss = 0
    for i in range(pcnt.shape[0]):
        num = np.min([pcnt[i].item(), ncnt[i].item()])
        if num<=0:
            continue
        if i==0:
            if num==pcnt[i]:
                idx = torch.multinomial(nval[i,-ncnt[i]:], num, replacement=True)
                idx -= ncnt[i]
                loss = ((nx[i,ninc[i,idx]]-px[i, pinc[i,-num:]])+1).clamp(min=0).mean()
            else:
                idx = torch.multinomial(pval[i,-pcnt[i]:], num, replacement=True)
                idx -= pcnt[i]
                loss = ((nx[i,ninc[i,-num:]]-px[i, pinc[i,idx]])+1).clamp(min=0).mean()
        else:
            if num==pcnt[i]:
                idx = torch.multinomial(nval[i,-ncnt[i]:], num, replacement=True)
                idx -= ncnt[i]
                loss += ((nx[i,ninc[i,idx]]-px[i, pinc[i,-num:]])+1).clamp(min=0).mean()
            else:
                idx = torch.multinomial(pval[i,-pcnt[i]:], num, replacement=True)
                idx -= pcnt[i]
                loss += ((nx[i,ninc[i,-num:]]-px[i, pinc[i,idx]])+1).clamp(min=0).mean()
    return loss


def UserPreferSum(e, v, u, invU):
    s = [torch.finfo(torch.float32).eps for i in u]
    for i in range(e.shape[1]):
        if e[0,i] in u:
            s[invU[e[0,i]]] += v[i]
    return s

class Pretrain_loss_train(nn.Module):
    def __init__(self, device=None):
        super(Pretrain_loss_train, self).__init__()
        self.device = device
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, KUU_adj, VUU_adj, Vuser, Vitem, Kuser, Kitem, Vid, preference, Vpreference, structure):
        Vu_adj = torch.matmul(Vuser, Vuser.T)
        tmp1 = VUU_adj.clone()
        tmp1[tmp1>0] = 1
        tmp1[tmp1<0] = 0
        e1 =  torch.eye(VUU_adj.shape[0])
        if tmp1.is_cuda:
            e1 = e1.to(tmp1.device)
        tmp1 = (-(tmp1-1))-e1
        Vu = contrast_loss(VUU_adj, tmp1, Vu_adj, Vu_adj)

        Ku_adj = torch.matmul(Kuser, Kuser.T)
        tmp2 = KUU_adj.clone()
        tmp2[tmp2>0] = 1
        tmp2[tmp2<0] = 0
        e2 = torch.eye(KUU_adj.shape[0])
        if tmp2.is_cuda:
            e2 = e2.to(tmp2.device)
        tmp2 = (-(tmp2-1))-e2
        tmp2 = tmp2[:Vu_adj.shape[0],-3:]
        Ku = contrast_loss(KUU_adj[:Vu_adj.shape[0], :KUU_adj.shape[1]-3], tmp2, Ku_adj[:Vu_adj.shape[0], :Ku_adj.shape[1]-3], Ku_adj[:Vu_adj.shape[0],-3:])
        
        Vi_adj = torch.matmul(Vuser, Vitem.T)
        VUI_adj = Vpreference.T
        tmp3 = VUI_adj.clone()
        tmp3[tmp3>0] = 1
        tmp3[tmp3<0] = 0
        tmp3 = (-(VUI_adj-1))
        Vi = contrast_loss(VUI_adj, tmp3, Vi_adj, Vi_adj)

        Ki_adj = torch.matmul(Kuser[:5,:], Kitem.T)
        KUI_adj = preference.clone().T
        tmp4 = KUI_adj.clone()
        tmp4[tmp4>0] = 1
        tmp4[tmp4<0] = 0
        tmp4 = (-(KUI_adj-1))
        Ki = contrast_loss(KUI_adj, tmp4, Ki_adj, Ki_adj)

        ii_adj = torch.matmul(Kitem, Vitem.T)
        ii = torch.abs(ii_adj-structure[:, :ii_adj.shape[1]]).mean()
        return Vu, Ku, Vi, Ki, ii


class Pretrain_loss_test(nn.Module):
    def __init__(self, device=None):
        super(Pretrain_loss_test, self).__init__()
        self.device = device
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, KUU_adj, VUU_adj, Vuser, Vitem, Kuser, Kitem, Vid, preference, Vpreference, structure):
        Vu_adj = torch.matmul(Vuser, Vuser.T)
        e1 = torch.eye(VUU_adj.shape[0])
        if VUU_adj.is_cuda:
            e1 = e1.to(VUU_adj.device)
        tmp1 = -(VUU_adj-1)-e1
        Vu = contrast_loss(VUU_adj, tmp1, Vu_adj, Vu_adj)

        Ku_adj = torch.matmul(Kuser, Kuser.T)
        e2 = torch.eye(KUU_adj.shape[0])
        if KUU_adj.is_cuda:
            e2 = e2.to(KUU_adj.device)
        tmp2 = (-(KUU_adj-1))-e2
        tmp2 = tmp2[:Vu_adj.shape[0],-3:]
        Ku = contrast_loss(KUU_adj[:Vu_adj.shape[0], :KUU_adj.shape[1]-3], tmp2, Ku_adj[:Vu_adj.shape[0], :Ku_adj.shape[1]-3], Ku_adj[:Vu_adj.shape[0],-3:])
        
        Vi_adj = torch.matmul(Vuser, Vitem.T)
        VUI_adj = preference[Vid[5:],:].T
        tmp3 = VUI_adj.clone()
        tmp3[tmp3>0] = 1
        tmp3 = -(VUI_adj-1)
        Vi = contrast_loss(VUI_adj, tmp3, Vi_adj, Vi_adj)

        Ki_adj = torch.matmul(Kuser[:5,:], Kitem.T)
        KUI_adj = preference.clone().T
        tmp4 = KUI_adj.clone()
        tmp4[tmp4>0] = 1
        tmp4 = (-(KUI_adj-1))
        Ki = contrast_loss(KUI_adj, tmp4, Ki_adj, Ki_adj)

        ii_adj = torch.matmul(Kitem, Vitem.T)
        ii = torch.abs(ii_adj-structure[:, :ii_adj.shape[1]]).mean()

        _, Recommend_i = torch.max(Vi_adj, dim=1)
        _, Replace_i = torch.min(Vi_adj, dim=1)
        _, Substitute_i = torch.max(Ki_adj, dim=1)
        Recommend = 0
        Replace = 0
        Substitute = 0
        for i in range(5):
            if Vid[i]==-1:
                break
            Recommend += preference[Vid[Recommend_i[i]+5], i]
            Replace += preference[Vid[Replace_i[i]+5], i]
            Substitute += preference[Substitute_i[i], i]

        return Vu, Ku, Vi, Ki, ii, Recommend, Replace, Substitute

class Predict_loss_train(nn.Module):
    def __init__(self, max_group_member):
        super(Predict_loss_train, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.max_group_member = max_group_member
        self.r = nn.ReLU(inplace=True)

    def forward(self, Recommended_m, Substitute_m, ItemGroups_m, Vid, VUU, KUU, Vscore, Kscore, II):
        # Vscore #UV
        # Kscore #UK
        # Recommended_m #UV
        # Substitute_m #UK
        # II #KV
        Vid = Vid[self.max_group_member:]

        recommend = Vscore*Recommended_m
        recommend = torch.sum(recommend, dim=1)
        
        substitute = Kscore*Substitute_m
        substitute = torch.sum(substitute, dim=1)

        display = Substitute_m #UK
        mask = torch.matmul(Substitute_m, Substitute_m.T)
        bias = torch.eye(Vscore.shape[0])
        if mask.is_cuda:
            bias = bias.to(mask.device)
        mask = mask*(-bias+1)
        social = KUU[:mask.shape[0], :mask.shape[1]]*mask
        social = torch.sum(social, dim=1)
        
        II = II[:,:Recommended_m.shape[1]]
        haptic = torch.sum(torch.matmul(Recommended_m, II.T)*Substitute_m, dim=1)
        
        return recommend, substitute, social, haptic

class Predict_loss_test(nn.Module):
    def __init__(self, max_group_member):
        super(Predict_loss_test, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.max_group_member = max_group_member

    def forward(self, Recommended_m, Substitute_m, ItemGroups_m, Vid, VUU, KUU, Vscore, Kscore, preference, structure):
        _, Recommended_i = torch.topk(Recommended_m, 1, dim=1, largest=True, sorted=True)
        _, Substitute_i = torch.topk(Substitute_m, 1, dim=1, largest=True, sorted=True)
        Vid = Vid[self.max_group_member:]
        Tprefer = torch.zeros(Recommended_m.shape[0], 1)
        Tsocial = torch.zeros(Recommended_m.shape[0], 1)
        Thaptic = torch.zeros(Recommended_m.shape[0], 1)
        preference = preference.cpu()
        structure = structure.cpu()
        Substitute_i = Substitute_i.cpu()
        Recommended_i = Recommended_i.cpu()
        Vid = Vid.cpu()
        Tprefer = Tprefer.cpu()
        Tsocial = Tsocial.cpu()
        Thaptic = Thaptic.cpu()

        for i in range(Tprefer.shape[0]):
            for j in range(Thaptic.shape[1]):
                Thaptic[i,j] = structure[Substitute_i[i, j], Recommended_i[i, j]]
                Tprefer[i,j] = preference[Substitute_i[i, j], i]-preference[Vid[Recommended_i[i, j]], i]

                Tsocial[i,j] = 0
                for k in range(Tprefer.shape[0]):
                    if k==i:
                        continue
                    if Substitute_i[k, j]==Substitute_i[i, j]:
                        Tsocial[i,j] += np.min([preference[Substitute_i[k, j], k], preference[Substitute_i[i, j], i]])

        return Tprefer, Tsocial, Thaptic
