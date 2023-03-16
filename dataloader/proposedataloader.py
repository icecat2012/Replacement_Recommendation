import torch.utils.data as data
from datetime import datetime
from torchvision import transforms
import random
from PIL import Image
import os
import torch
import numpy as np
import pickle
import time

class time_analysis(object):
    def __init__(self):
        self.s = 0.0
        self.duration = 0.0
    def start(self):
        self.s = time.time()
    def end(self):
        self.duration = self.duration + (time.time() - self.s)
        print(time.time(), self.s)

    
def loader_(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def initial_load(dataset_name):
    with open('../data/{}/data/user_merged_{}2code.pkl'.format(dataset_name, dataset_name[3:].lower()), 'rb') as f:
        Umap = pickle.load(f)
    with open('../data/{}/data/item_{}2code.pkl'.format(dataset_name, dataset_name[3:].lower()), 'rb') as f:
        Imap = pickle.load(f)
    with open('../data/{}/data/item_embedding.pkl'.format(dataset_name), 'rb') as f:
        feature = pickle.load(f)
    Vitem = torch.cat([torch.unsqueeze(feature[k], 0) for k in range(len(Imap))], dim=0)
    with open('../data/{}/data/item_embedding_restnet.pkl'.format(dataset_name), 'rb') as f:
        feature = pickle.load(f)
    Kitem = torch.cat([torch.unsqueeze(feature[k], 0) for k in range(len(Imap))], dim=0)
    with open('../data/{}/data/user_merged_groupcode2membercode.pkl'.format(dataset_name), 'rb') as f:
        group_map = pickle.load(f)
    Gmap = dict()
    for i in range(len(group_map)):
        Gmap[i] = sorted(group_map[i])

    with open('../data/{}/data/proposed/Vuser_embedding.pkl'.format(dataset_name), 'rb') as f:
        user_x = pickle.load(f)
    Vuser = torch.cat([user_x[k] for k in range(len(Umap))], dim=0)
    with open('../data/{}/data/proposed/Kuser_embedding.pkl'.format(dataset_name), 'rb') as f:
        user_x = pickle.load(f)
    Kuser = torch.cat([user_x[k] for k in range(len(Umap))], dim=0)

    with open('../data/{}/data/scene2code.pkl'.format(dataset_name), 'rb') as f:
        scene2item = pickle.load(f) #list

    return Umap, Imap, Vitem, Kitem, Vuser, Kuser, Gmap, scene2item

def get_graph_edge(dataset_name, mode='True'):
    with open('../data/{}/data/proposed/True_item_item.pkl'.format(dataset_name), 'rb') as f:
        (IIedge, IIvalue) = pickle.load(f)
    with open('../data/{}/data/proposed/{}_friend.pkl'.format(dataset_name, mode), 'rb') as f:
        (UUedge, UUvalue) = pickle.load(f)
    with open('../data/{}/data/proposed/{}_preference.pkl'.format(dataset_name, mode), 'rb') as f:
        (UIedge, UIvalue) = pickle.load(f)
    return IIedge, IIvalue, UUedge, UUvalue, UIedge, UIvalue


def get_index(dataset_name, mode='train'):
    with open('../data/{}/data/proposed/{}/index.pkl'.format(dataset_name, mode), 'rb') as f:
        index = pickle.load(f)
    return index

def check_graph_index(x, e, v, label):
    print(label)
    n = x.shape[0]
    if e.shape[1]!=v.shape[0]:
        print('len(e)!=len(v)', e.shape[1], v.shape[0])
        exit()
    for i in range(e.shape[1]):
        if e[0,i]>=n or e[1,i]>=n:
            print('e>=n', e[:,i], n)
            exit()

class TrainDataset(data.Dataset):
    def __init__(self, dataset_name, split_low, split_high, max_group_member=5):
        random.seed(datetime.now())

        self.dataset_name = dataset_name
        self.Umap, self.Imap, self.Vitem, self.Kitem, self.Vusers, self.Kusers, self.Gmap, self.scenes= initial_load(dataset_name)
        self.IIedge, self.IIvalue, self.UUedge, self.UUvalue, self.UIedge, self.UIvalue = get_graph_edge(dataset_name, mode='train')
        self.index = get_index(dataset_name, mode='train')
        
        
        self.len = len(self.index)
        self.max_group_member = max_group_member
        self.max_scene_item = 10

    def __getitem__(self, idx):
        group, scene = self.index[idx]
        Vitem = self.scenes[scene]
        user = self.Gmap[group]
        Vid = user.copy()
        for i in range(self.max_group_member-len(user)):
            Vid.append(-1)
        Kid = Vid.copy()
        
        Vuser = torch.cat([self.Vusers[user[i]].unsqueeze(0) if i<len(user) else torch.zeros((1,256)) for i in range(self.max_group_member)], dim=0)
        Kuser = torch.cat([self.Kusers[user[i]].unsqueeze(0) if i<len(user) else torch.zeros((1,256)) for i in range(self.max_group_member)], dim=0)

        Vgraph = torch.cat([torch.unsqueeze(self.Vitem[i,:], 0) for i in Vitem], dim=0)
        structure = torch.matmul(self.Vitem, Vgraph.clone().T)
        Vgraph = torch.cat([Vuser, Vgraph], dim=0)
        

        with open('../data/{}/data/proposed/{}/group{}_scene{}.txt'.format(self.dataset_name, 'train', group, scene), 'r') as f:
            UUidx = f.readline().strip('\n').split('\t')
            UUidx = [int(float(i)) for i in UUidx]
            UIidx = f.readline().strip('\n').split('\t')
            UIidx = [int(float(i)) for i in UIidx]
            pid = f.readline().strip('\n').split('\t')
            if len(pid)==1 and pid[0]=='':
                pid = []
            else:
                pid = [int(float(i)) for i in pid]
            nid = f.readline().strip('\n').split('\t')
            nid = [int(float(i)) for i in nid]
        
        if len(pid)>20:
            random.shuffle(pid)
            pid = pid[:20]
        random.shuffle(nid)
        nid = nid[:20]

        Tmap = dict()
        Tcnt = 0
        for u in user:
            Tmap[u] = Tcnt
            Tcnt+=1
        if len(Tmap)<self.max_group_member:
            Tmap[-1] = Tcnt
            Tcnt+=1
        for u in nid:
            Tmap[u] = Tcnt
            Tcnt+=1
        # Tcnt += (self.max_group_member - len(user))
        for u in pid:
            Tmap[u] = Tcnt
            Tcnt+=1

        preference = torch.zeros(self.Kitem.shape[0], self.max_group_member)
        Kuser = torch.cat([Kuser.clone(), self.Kusers[nid], self.Kusers[pid]], dim=0)
        Kedge = [list(), list()]
        Kvalue = list()
        for i in UUidx:
            if self.UUedge[0,i] in Tmap and self.UUedge[1,i] in Tmap:#self.UUvalue[i]>0.23
                Kedge[0].append(Tmap[self.UUedge[0,i]])
                Kedge[1].append(Tmap[self.UUedge[1,i]])
                Kvalue.append(self.UUvalue[i])
        for i in UIidx:
            if self.UIedge[0,i] in Tmap:
                Kedge[0].append(Tmap[self.UIedge[0,i]])
                Kedge[1].append(self.UIedge[1,i]+Kuser.shape[0])
                Kvalue.append(self.UIvalue[i])
                if Tmap[self.UIedge[0,i]]<len(user):
                    preference[self.UIedge[1,i], Tmap[self.UIedge[0,i]]] = self.UIvalue[i]

        cnt = preference.clone()
        cnt[cnt>0] = 1
        cnt[cnt<0] = 0
        cnt = torch.sum(cnt, dim=1)
        for i in range(preference.shape[0]):
            for j in range(preference.shape[1]):
                if preference[i, j]==0 and cnt[i]>0:
                    preference[i, j] = 0.01*cnt[i]
        

        Kgraph = torch.cat([Kuser, self.Kitem], dim=0)
        
        Kid.extend(nid)
        Kid.extend(pid)
        Kid.extend([i for i in range(len(self.Imap))])
        Vid.extend(Vitem)

        Kitem = list()
        for i in range(len(self.Imap)):
            if i not in Vitem:
                Kitem.append(i)
        ################################ make vedge #################################
        invUser = dict()
        cnt=0
        for u in user:
            invUser[u] = cnt
            cnt+=1
        invItem = dict()
        cnt=0
        for i in Vitem:
            invItem[i] = cnt
            cnt+=1

        Vedge = [list(), list()]
        Vvalue = list()
        user_t = set(user)
        Vitem_t = set(Vitem)

        VUU_adj = torch.eye(self.max_group_member)
        KUU_adj = torch.eye(Tcnt)
        Vpreference = torch.zeros(len(Vitem_t), self.max_group_member)
        for i in range(self.UUedge.shape[1]):
            if self.UUedge[0,i] in user_t and self.UUedge[1,i] in user_t:#UUvalue[i]>0.23 and 
                Vedge[0].append(invUser[self.UUedge[0,i]])
                Vedge[1].append(invUser[self.UUedge[1,i]])
                Vvalue.append(self.UUvalue[i])
                if self.UUedge[0,i]!=self.UUedge[1,i]:
                    VUU_adj[invUser[self.UUedge[0,i]], invUser[self.UUedge[1,i]]] = self.UUvalue[i]
                    VUU_adj[invUser[self.UUedge[1,i]], invUser[self.UUedge[0,i]]] = self.UUvalue[i]
            if self.UUedge[0,i] in Tmap and self.UUedge[1,i] in Tmap and self.UUedge[0,i]!=self.UUedge[1,i]:
                KUU_adj[Tmap[self.UUedge[0,i]], Tmap[self.UUedge[1,i]]] = self.UUvalue[i]
                KUU_adj[Tmap[self.UUedge[1,i]], Tmap[self.UUedge[0,i]]] = self.UUvalue[i]
        for i in range(self.UIedge.shape[1]):
            if self.UIedge[0,i] in user_t and self.UIedge[1,i] in Vitem_t:
                Vedge[0].append(invUser[self.UIedge[0,i]])
                Vedge[1].append(invItem[self.UIedge[1,i]]+self.max_group_member)
                Vvalue.append(self.UIvalue[i])
                Vpreference[invItem[self.UIedge[1,i]], invUser[self.UIedge[0,i]]] = self.UIvalue[i]

        cnt = Vpreference.clone()
        cnt[cnt>0] = 1
        cnt[cnt<0] = 0
        cnt = torch.sum(cnt, dim=1)
        for i in range(Vpreference.shape[0]):
            for j in range(Vpreference.shape[1]):
                if Vpreference[i, j]==0 and cnt[i]>0:
                    Vpreference[i, j] = 0.01*cnt[i]

        KUU_adj = KUU_adj-torch.eye(Tcnt)
        VUU_adj = VUU_adj-torch.eye(self.max_group_member)

        # print(Vitem, user)
        # check_graph_index(Vgraph, Vedge, Vvalue, '@V')
        # check_graph_index(Kgraph, np.array(Kedge), np.array(Kvalue), '@K')

        return Vgraph, torch.tensor(Vedge), torch.tensor(Vvalue), Kgraph, np.array(Kedge), np.array(Kvalue), np.array(Vid), np.array(Kid), np.array(Vitem), np.array(Kitem), preference, Vpreference, structure, KUU_adj, VUU_adj

    def __len__(self):
        return self.len

class TestDataset(data.Dataset):
    def __init__(self, dataset_name, split_low, split_high, max_group_member=5):
        random.seed(2147483647)

        self.dataset_name = dataset_name
        self.Umap, self.Imap, self.Vitem, self.Kitem, self.Vusers, self.Kusers, self.Gmap, self.scenes = initial_load(dataset_name)
        self.IIedge, self.IIvalue, self.UUedge, self.UUvalue, self.UIedge, self.UIvalue = get_graph_edge(dataset_name, mode='True')
        self.index = get_index(dataset_name, mode='test')
        random.shuffle(self.index)
        
        self.len = len(self.index)
        self.max_group_member = max_group_member
        self.max_scene_item = 10

    def __getitem__(self, idx):
        group, scene = self.index[idx]
        Vitem = self.scenes[scene]
        user = self.Gmap[group]
        Vid = user.copy()
        for i in range(self.max_group_member-len(user)):
            Vid.append(-1)
        Kid = Vid.copy()
        
        Vuser = torch.cat([self.Vusers[user[i]].unsqueeze(0) if i<len(user) else torch.zeros((1,256)) for i in range(self.max_group_member)], dim=0)
        Kuser = torch.cat([self.Kusers[user[i]].unsqueeze(0) if i<len(user) else torch.zeros((1,256)) for i in range(self.max_group_member)], dim=0)
        
        Vgraph = torch.cat([torch.unsqueeze(self.Vitem[i,:], 0) for i in Vitem], dim=0)
        structure = torch.matmul(self.Vitem, Vgraph.clone().T)
        
        Vgraph = torch.cat([Vuser, Vgraph], dim=0)

        with open('../data/{}/data/proposed/{}/group{}_scene{}.txt'.format(self.dataset_name, 'test', group, scene), 'r') as f:
            UUidx = f.readline().strip('\n').split('\t')
            UUidx = [int(float(i)) for i in UUidx]
            UIidx = f.readline().strip('\n').split('\t')
            UIidx = [int(float(i)) for i in UIidx]
            pid = f.readline().strip('\n').split('\t')
            if len(pid)==1 and pid[0]=='':
                pid = []
            else:
                pid = [int(float(i)) for i in pid]
            nid = f.readline().strip('\n').split('\t')
            nid = [int(float(i)) for i in nid]
        
        if len(pid)>20:
            random.shuffle(pid)
            pid = pid[:20]
        random.shuffle(nid)
        nid = nid[:20]

        Tmap = dict()
        Tcnt = 0
        for u in user:
            Tmap[u] = Tcnt
            Tcnt+=1
        if len(Tmap)<self.max_group_member:
            Tmap[-1] = Tcnt
            Tcnt+=1
        for u in nid:
            Tmap[u] = Tcnt
            Tcnt+=1
        for u in pid:
            Tmap[u] = Tcnt
            Tcnt+=1
        
        preference = torch.zeros(self.Kitem.shape[0], self.max_group_member)
        Kuser = torch.cat([Kuser.clone(), self.Kusers[nid], self.Kusers[pid]], dim=0)
        Kedge = [list(), list()]
        Kvalue = list()
        for i in UUidx:
            if self.UUedge[0,i] in Tmap and self.UUedge[1,i] in Tmap:#self.UUvalue[i]>0.23 
                Kedge[0].append(Tmap[self.UUedge[0,i]])
                Kedge[1].append(Tmap[self.UUedge[1,i]])
                Kvalue.append(self.UUvalue[i])
        for i in UIidx:
            if self.UIedge[0,i] in Tmap:
                Kedge[0].append(Tmap[self.UIedge[0,i]])
                Kedge[1].append(self.UIedge[1,i]+Kuser.shape[0])
                Kvalue.append(self.UIvalue[i])
                if Tmap[self.UIedge[0,i]]<len(user):
                    preference[self.UIedge[1,i], Tmap[self.UIedge[0,i]]] = self.UIvalue[i]


        cnt = preference.clone()
        cnt[cnt>0] = 1
        cnt[cnt<0] = 0
        cnt = torch.sum(cnt, dim=1)
        for i in range(preference.shape[0]):
            for j in range(preference.shape[1]):
                if preference[i, j]==0 and cnt[i]>0:
                    preference[i, j] = 0.01*cnt[i]

        Kgraph = torch.cat([Kuser, self.Kitem], dim=0)
        
        Kid.extend(nid)
        Kid.extend(pid)
        Kid.extend([i for i in range(len(self.Imap))])
        Vid.extend(Vitem)

        Kitem = list()
        for i in range(len(self.Imap)):
            if i not in Vitem:
                Kitem.append(i)

################################ make vedge #################################
        invUser = dict()
        cnt=0
        for u in user:
            invUser[u] = cnt
            cnt+=1
        invItem = dict()
        cnt=0
        for i in Vitem:
            invItem[i] = cnt
            cnt+=1

        Vedge = [list(), list()]
        Vvalue = list()
        user_t = set(user)
        Vitem_t = set(Vitem)

        VUU_adj = torch.eye(self.max_group_member)
        KUU_adj = torch.eye(Tcnt)
        Vpreference = torch.zeros(len(Vitem_t), self.max_group_member)
        for i in range(self.UUedge.shape[1]):
            if self.UUedge[0,i] in user_t and self.UUedge[1,i] in user_t:#UUvalue[i]>0.23 and 
                Vedge[0].append(invUser[self.UUedge[0,i]])
                Vedge[1].append(invUser[self.UUedge[1,i]])
                Vvalue.append(self.UUvalue[i])
                if self.UUedge[0,i]!=self.UUedge[1,i]:
                    VUU_adj[invUser[self.UUedge[0,i]], invUser[self.UUedge[1,i]]] = self.UUvalue[i]
                    VUU_adj[invUser[self.UUedge[1,i]], invUser[self.UUedge[0,i]]] = self.UUvalue[i]
            if self.UUedge[0,i] in Tmap and self.UUedge[1,i] in Tmap and self.UUedge[0,i]!=self.UUedge[1,i]:
                KUU_adj[Tmap[self.UUedge[0,i]], Tmap[self.UUedge[1,i]]] = self.UUvalue[i]
                KUU_adj[Tmap[self.UUedge[1,i]], Tmap[self.UUedge[0,i]]] = self.UUvalue[i]
        for i in range(self.UIedge.shape[1]):
            if self.UIedge[0,i] in user_t and self.UIedge[1,i] in Vitem_t:
                Vedge[0].append(invUser[self.UIedge[0,i]])
                Vedge[1].append(invItem[self.UIedge[1,i]]+self.max_group_member)
                Vvalue.append(self.UIvalue[i])
                Vpreference[invItem[self.UIedge[1,i]], invUser[self.UIedge[0,i]]] = self.UIvalue[i]

        cnt = Vpreference.clone()
        cnt[cnt>0] = 1
        cnt[cnt<0] = 0
        cnt = torch.sum(cnt, dim=1)
        for i in range(Vpreference.shape[0]):
            for j in range(Vpreference.shape[1]):
                if Vpreference[i, j]==0 and cnt[i]>0:
                    Vpreference[i, j] = 0.01*cnt[i]
        KUU_adj = KUU_adj-torch.eye(Tcnt)
        VUU_adj = VUU_adj-torch.eye(self.max_group_member)
        # print(Vitem, user)
        # check_graph_index(Vgraph, Vedge, Vvalue, '@V')
        # check_graph_index(Kgraph, np.array(Kedge), np.array(Kvalue), '@K')
        return Vgraph, torch.tensor(Vedge), torch.tensor(Vvalue), Kgraph, np.array(Kedge), np.array(Kvalue), np.array(Vid), np.array(Kid), np.array(Vitem), np.array(Kitem), preference, Vpreference, structure, KUU_adj, VUU_adj

    def __len__(self):
        return self.len

class OutputDataset(data.Dataset):
    def __init__(self, dataset_name, split_low, split_high, max_group_member=5, max_scene_item=17):
        self.dataset_name = dataset_name
        self.Umap, self.Imap, self.Vitem, self.Kitem, self.Vusers, self.Kusers, self.Gmap, self.scenes = initial_load(dataset_name)
        self.IIedge, self.IIvalue, self.UUedge, self.UUvalue, self.UIedge, self.UIvalue = get_graph_edge(dataset_name, mode='True')
        self.index = get_index(dataset_name, mode='test')
        self.len = len(self.index)
        self.max_group_member = max_group_member
        self.max_scene_item = max_scene_item

    def __getitem__(self, idx):
        group, scene = self.index[idx]
        Vitem = self.scenes[scene]
        user = self.Gmap[group]
        Vid = user.copy()
        for i in range(self.max_group_member-len(user)):
            Vid.append(-1)
        Kid = Vid.copy()
        
        Vuser = torch.cat([torch.unsqueeze(self.Vusers[user[i]], 0) if i<len(user) else torch.zeros((1,256)) for i in range(self.max_group_member)], dim=0)
        Kuser = torch.cat([torch.unsqueeze(self.Kusers[user[i]], 0) if i<len(user) else torch.zeros((1,256)) for i in range(self.max_group_member)], dim=0)
        
        Vgraph = torch.cat([torch.unsqueeze(self.Vitem[i,:], 0) for i in Vitem], dim=0)
        Vgraph = torch.cat([Vuser, Vgraph], dim=0)

        with open('../data/{}/data/proposed/{}/group{}_scene{}.txt'.format(self.dataset_name, 'test', group, scene), 'r') as f:
            UUidx = f.readline().strip('\n').split('\t')
            UUidx = [int(float(i)) for i in UUidx]
            UIidx = f.readline().strip('\n').split('\t')
            UIidx = [int(float(i)) for i in UIidx]
            pid = f.readline().strip('\n').split('\t')
            pid = [int(float(i)) for i in pid]
            nid = f.readline().strip('\n').split('\t')
            nid = [int(float(i)) for i in nid]
        
        if len(pid)>20:
            random.shuffle(pid)
            pid = pid[:20]
        random.shuffle(nid)
        nid = nid[:20]

        Tmap = dict()
        Tcnt = 0
        for u in user:
            Tmap[u] = Tcnt
            Tcnt+=1
        if len(Tmap)<self.max_group_member:
            Tmap[-1] = Tcnt
            Tcnt+=1
        for u in nid:
            Tmap[u] = Tcnt
            Tcnt+=1
        for u in pid:
            Tmap[u] = Tcnt
            Tcnt+=1
        
        Kuser = torch.cat([Kuser.clone(), self.Kusers[nid], self.Kusers[pid]], dim=0)
        Kedge = [list(), list()]
        Kvalue = list()
        for i in UUidx:
            if self.UUedge[0,i] in Tmap and self.UUedge[1,i] in Tmap:#self.UUvalue[i]>0.23 
                Kedge[0].append(Tmap[self.UUedge[0,i]])
                Kedge[1].append(Tmap[self.UUedge[1,i]])
                Kvalue.append(self.UUvalue[i])
        for i in UIidx:
            if self.UIedge[0,i] in Tmap:
                Kedge[0].append(Tmap[self.UIedge[0,i]])
                Kedge[1].append(self.UIedge[1,i]+Kuser.shape[0])
                Kvalue.append(self.UIvalue[i])

        Kgraph = torch.cat([Kuser, self.Kitem], dim=0)
        
        Kid.extend(nid)
        Kid.extend(pid)
        Kid.extend([i for i in range(len(self.Imap))])
        Vid.extend(Vitem)

        Kitem = list()
        for i in range(len(self.Imap)):
            if i not in Vitem:
                Kitem.append(i)

################################ make vedge #################################
        invUser = dict()
        cnt=0
        for u in user:
            invUser[u] = cnt
            cnt+=1
        invItem = dict()
        cnt=0
        for i in Vitem:
            invItem[i] = cnt
            cnt+=1

        Vedge = [list(), list()]
        Vvalue = list()
        user_t = set(user)
        Vitem_t = set(Vitem)

        VUU_adj = torch.eye(self.max_group_member)
        KUU_adj = torch.eye(Tcnt)
        Vpreference = torch.zeros(len(Vitem_t), self.max_group_member)
        for i in range(self.UUedge.shape[1]):
            if self.UUedge[0,i] in user_t and self.UUedge[1,i] in user_t:#UUvalue[i]>0.23 and 
                Vedge[0].append(invUser[self.UUedge[0,i]])
                Vedge[1].append(invUser[self.UUedge[1,i]])
                Vvalue.append(self.UUvalue[i])
        for i in range(self.UIedge.shape[1]):
            if self.UIedge[0,i] in user_t and self.UIedge[1,i] in Vitem_t:
                Vedge[0].append(invUser[self.UIedge[0,i]])
                Vedge[1].append(invItem[self.UIedge[1,i]]+self.max_group_member)
                Vvalue.append(self.UIvalue[i])

        return Vgraph, torch.tensor(Vedge), torch.tensor(Vvalue), Kgraph, np.array(Kedge), np.array(Kvalue), np.array(Vid), np.array(Kid), np.array(Vitem), np.array(Kitem)

    def __len__(self):
        return self.len
