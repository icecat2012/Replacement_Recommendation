import numpy as np
from dataloader.proposedataloader import OutputDataset 
from torch.utils.data import DataLoader
import torch
import os
import pickle
import torch.multiprocessing
import multiprocessing
from model.model import Yolo_GCN, Group_Recommendation

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Graph_pkg(object):
    def __init__(self, x, e, v, nid=None, b=None, device='cpu'):
        super(Graph_pkg, self).__init__()
        self.x = x
        self.edge = e
        self.value = v
        self.batch = b
        self.nid = nid
        if b is None:
            self.batch = torch.zeros(x.shape[0]).type(torch.LongTensor).to(device=device)
        if nid is None:
            self.nid = torch.tensor([i for i in range(x.shape[0])]).type(torch.LongTensor).to(device=device)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def tostring(inlist):
    s = ''
    for i in inlist:
        s = s + ', ' + str(int(i))
    return '[' + s[2:] + ']'



def our(test_loader, modelF, model, path):
    modelF.eval()
    model.eval()
    Batch_test_error = AverageMeter()
    Batch_test_error_s = AverageMeter()
    Batch_test_error_c = AverageMeter()
    Batch_test_error_i = AverageMeter()
    with torch.no_grad():
        with open('{}.txt'.format(path), 'w') as f:
            for bth, data in enumerate(test_loader, 0):
                Vgraph, Vedge, Vvalue, Kgraph, Kedge, Kvalue, Vid, Kid, Vitem, Kitem = data

                Vgraph = Vgraph.squeeze(0).clone()
                Vedge = Vedge.squeeze(0).clone()
                Vvalue = Vvalue.squeeze(0).clone()
                Kgraph = Kgraph.squeeze(0).clone()
                Kedge = Kedge.squeeze(0).clone()
                Kvalue = Kvalue.squeeze(0).clone()
                Vitem = Vitem.squeeze(0).clone()
                Kitem = Kitem.squeeze(0).clone()

                VGraph = Graph_pkg(Vgraph, Vedge, Vvalue.type(torch.float), Vid)
                KGraph = Graph_pkg(Kgraph, Kedge, Kvalue.type(torch.float), Kid)
                Vuser, Vitem, Kuser, Kitem = modelF(VGraph, KGraph, Vitem, Kitem)
                Recommended_m, Substitute_m, ItemGroups_m, VUU, KUU, Vscore, Kscore, II = model(Vuser, Vitem, Kuser[:5,:], Kitem)

                _, Recommended_i = torch.topk(Recommended_m, 1, dim=1, largest=True, sorted=True)
                _, Substitute_i = torch.topk(Substitute_m, 1, dim=1, largest=True, sorted=True)
                Vid = Vid[0]
                if Vid[4]==-1:
                    users = Vid[:4]
                else:
                    users = Vid[:5]
                items = Vid[5:]
                users = users.numpy()
                items = items.numpy()

                Vid = Vid.tolist()
                Recommended_i = Recommended_i.reshape(-1).tolist()
                Substitute_i = Substitute_i.reshape(-1).tolist()
                for i in range(len(Recommended_i)):
                    Recommended_i[i] = items[Recommended_i[i]]
                rep = Recommended_i
                sub = Substitute_i
                
                f.write('{},{},{},{}\n'.format(tostring(users), tostring(items), tostring(rep), tostring(sub)))


def main(path='./save0/Epoch_0_Batch_501_Val_1.876796'):
    device = 'cuda:0'
    batchsize = 1
    n_classes = 400
    max_group_member = 5
    max_scene_item = 10


    test_dataset = OutputDataset('NewAmazon', split_low=0.2, split_high=1.1)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=int(8))

    modelF = Yolo_GCN(n_classes=n_classes, inference=False, n_users=len(test_dataset.Umap), n_items=len(test_dataset.Imap), max_group_member=max_group_member, max_scene_item=max_scene_item)
    pretrained1 = torch.load('{}F.pth'.format(path), map_location=torch.device(device))
    modelF.load_state_dict(pretrained1)
    model = Group_Recommendation(n_classes=n_classes, max_group_member=max_group_member, max_scene_item=max_scene_item)
    pretrained2 = torch.load('{}.pth'.format(path), map_location=torch.device(device))
    model.load_state_dict(pretrained2)
    our(test_loader, modelF, model, path)


if __name__=='__main__':
    folders = ['save0', 'save1', 'save2', 'save3', 'save4']
    files = list()
    for fld in folders:
        fs = os.listdir(fld)
        for f in fs:
            n = f.rsplit('.',1)
            if n[1]=='pth' and n[0][-1]=='F':
                files.append('{}/{}'.format(fld, n[0][:-1]))

#    num = multiprocessing.cpu_count()//8
    for i in range(0,len(files),1):
        main(files[i])
#        jobs = list()
#        for j in range(num):
#            if i+j>=len(files):
#                break
            # print(files[i+j])
#            ps = multiprocessing.Process(target=main, args=(files[i+j],))
#            jobs.append(ps)
#            ps.start()

#        for proc in jobs:
#            proc.join()
