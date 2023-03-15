import numpy as np
from model.model import Yolo_GCN, Group_Recommendation
from dataloader.proposedataloader import TrainDataset
from dataloader.proposedataloader import TestDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss.yolov4_loss import Yolo_loss
from loss.graph_loss import Pretrain_loss_train, Pretrain_loss_test, Predict_loss_train, Predict_loss_test
import os
import pickle
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

def inference_train(optimizer, modelF, model, max_group_member, VGraph, KGraph, Vitem, Kitem, KUU_adj, VUU_adj, Vid, preference, Vpreference, structure, criterion1, criterion2):
    modelF.zero_grad()
    model.zero_grad()
    Vuser, Vitem, Kuser, Kitem = modelF(VGraph, KGraph, Vitem, Kitem)
    Recommended_m, Substitute_m, ItemGroups_m, VUU, KUU, Vscore, Kscore, II = model(Vuser, Vitem, Kuser[:max_group_member,:], Kitem)
    
    Vu, Ku, Vi, Ki, ii = criterion1(KUU_adj[0], VUU_adj[0], Vuser, Vitem, Kuser, Kitem, Vid, preference, Vpreference, structure)
    Vs = preference[Vid[5:],:]
    recommend, substitute, social, haptic = criterion2(Recommended_m, Substitute_m, ItemGroups_m, Vid, VUU_adj[0], KUU_adj[0], Vs.T, preference.T, structure)
    supervised = Vu + Ku + Vi + Ki + ii
    prefer = (substitute-recommend).mean()
    substitute = substitute.mean()
    recommend = recommend.mean()
    social = social.mean()
    haptic = haptic.mean()
    loss = supervised-(prefer+social+haptic)

    loss.backward()

    optimizer.step()
    return modelF, model, loss.item()

def inference_test(max_group_member, modelF, model, criterion1, criterion2, VGraph, KGraph, Vitem, Kitem, KUU_adj, VUU_adj, Vid, preference, Vpreference, structure):
    Vuser, Vitem, Kuser, Kitem = modelF(VGraph, KGraph, Vitem, Kitem)
    Recommended_m, Substitute_m, ItemGroups_m, VUU, KUU, Vscore, Kscore, II = model(Vuser, Vitem, Kuser[:max_group_member,:], Kitem)

    Vu, Ku, Vi, Ki, ii, Recommend, Replace, Substitute = criterion1(KUU_adj[0], VUU_adj[0], Vuser, Vitem, Kuser, Kitem, Vid, preference, Vpreference, structure)
    Tprefer, Tsocial, Thaptic = criterion2(Recommended_m, Substitute_m, ItemGroups_m, Vid, VUU, KUU, Vscore, Kscore, preference, structure)

    supervised = Vu + Ku + Vi + Ki + ii
    Tprefer = Tprefer.mean()
    Tsocial = Tsocial.mean()
    Thaptic = Thaptic.mean()
    loss = supervised-(Tprefer+Tsocial+Thaptic)
    return loss.item(), Recommend.item(), Replace.item(), Substitute.item(), Tprefer.item(), Tsocial.item(), Thaptic.item()

def save_latest(modelF, model, c):
    try:
        save_path = './save{}/Latest.pth'.format(c)
        torch.save(model.state_dict(), save_path)
        save_path = './save{}/LatestF.pth'.format(c)
        torch.save(modelF.state_dict(), save_path)
    except Exception as e:
        print('ez1', e)

def train(train_loader, test_loader, modelF0, model0, modelF1, model1, modelF2, model2, modelF3, model3, modelF4, model4, max_group_member, device, n_classes):
    optimizer0 = optim.Adam(list(modelF0.parameters())+list(model0.parameters()),lr=0.0001/max_group_member,betas=(0.9, 0.999),eps=1e-08)
    optimizer1 = optim.Adam(list(modelF1.parameters())+list(model1.parameters()),lr=0.0001/max_group_member,betas=(0.9, 0.999),eps=1e-08)
    optimizer2 = optim.Adam(list(modelF2.parameters())+list(model2.parameters()),lr=0.0001/max_group_member,betas=(0.9, 0.999),eps=1e-08)
    optimizer3 = optim.Adam(list(modelF3.parameters())+list(model3.parameters()),lr=0.0001/max_group_member,betas=(0.9, 0.999),eps=1e-08)
    optimizer4 = optim.Adam(list(modelF4.parameters())+list(model4.parameters()),lr=0.0001/max_group_member,betas=(0.9, 0.999),eps=1e-08)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)
    Total_error = [AverageMeter() for i in range(5)]
    criterion1 = Pretrain_loss_train()
    criterion_detail1 = Pretrain_loss_test()
    criterion2 = Predict_loss_train(max_group_member)
    criterion_detail2 = Predict_loss_test(max_group_member)
    
    best = [-999999 for i in range(5)]
    for epoch in range(6):
        Batch_train_error = [AverageMeter() for i in range(5)]

        for bth, data in enumerate(train_loader, 0):
            modelF0.train()
            model0.train()
            modelF1.train()
            model1.train()
            modelF2.train()
            model2.train()
            modelF3.train()
            model3.train()
            modelF4.train()
            model4.train()
            Vgraph, Vedge, Vvalue, Kgraph, Kedge, Kvalue, Vid, Kid, Vitem, Kitem, preference, Vpreference, structure, KUU_adj, VUU_adj = data
            
            Vgraph = Vgraph.squeeze(0).to(device=device)
            Vedge = Vedge.squeeze(0).to(device=device)
            Vvalue = Vvalue.squeeze(0).to(device=device)
            Vid = Vid.squeeze(0).to(device=device)
            Kgraph = Kgraph.squeeze(0).to(device=device)
            Kedge = Kedge.squeeze(0).to(device=device)
            Kvalue = Kvalue.squeeze(0).to(device=device)
            Kid = Kid.squeeze(0).to(device=device)
            Vitem = Vitem.squeeze(0).to(device=device)
            Kitem = Kitem.squeeze(0).to(device=device)
            preference = preference.squeeze(0).to(device=device)
            structure = structure.squeeze(0).to(device=device)
            Vpreference = Vpreference.squeeze(0).to(device=device)
            KUU_adj = KUU_adj.to(device=device)
            VUU_adj = VUU_adj.to(device=device)

            VGraph = Graph_pkg(Vgraph.clone(), Vedge.clone(), Vvalue.clone().type(torch.float).clone(), Vid.clone(), device=device)
            KGraph = Graph_pkg(Kgraph.clone(), Kedge.clone(), Kvalue.clone().type(torch.float).clone(), Kid.clone(), device=device)
            modelF0, model0, loss0 = inference_train(optimizer0, modelF0, model0, max_group_member, VGraph, KGraph, Vitem.clone(), Kitem.clone(), KUU_adj.clone(), VUU_adj.clone(), Vid.clone(), preference.clone(), Vpreference.clone(), structure.clone(), criterion1, criterion2)

            VGraph = Graph_pkg(Vgraph.clone(), Vedge.clone(), Vvalue.clone().type(torch.float).clone(), Vid.clone(), device=device)
            KGraph = Graph_pkg(Kgraph.clone(), Kedge.clone(), Kvalue.clone().type(torch.float).clone(), Kid.clone(), device=device)
            modelF1, model1, loss1 = inference_train(optimizer1, modelF1, model1, max_group_member, VGraph, KGraph, Vitem.clone(), Kitem.clone(), KUU_adj.clone(), VUU_adj.clone(), Vid.clone(), preference.clone(), Vpreference.clone(), structure.clone(), criterion1, criterion2)

            VGraph = Graph_pkg(Vgraph.clone(), Vedge.clone(), Vvalue.clone().type(torch.float).clone(), Vid.clone(), device=device)
            KGraph = Graph_pkg(Kgraph.clone(), Kedge.clone(), Kvalue.clone().type(torch.float).clone(), Kid.clone(), device=device)
            modelF2, model2, loss2 = inference_train(optimizer2, modelF2, model2, max_group_member, VGraph, KGraph, Vitem.clone(), Kitem.clone(), KUU_adj.clone(), VUU_adj.clone(), Vid.clone(), preference.clone(), Vpreference.clone(), structure.clone(), criterion1, criterion2)

            VGraph = Graph_pkg(Vgraph.clone(), Vedge.clone(), Vvalue.clone().type(torch.float).clone(), Vid.clone(), device=device)
            KGraph = Graph_pkg(Kgraph.clone(), Kedge.clone(), Kvalue.clone().type(torch.float).clone(), Kid.clone(), device=device)
            modelF3, model3, loss3 = inference_train(optimizer3, modelF3, model3, max_group_member, VGraph, KGraph, Vitem.clone(), Kitem.clone(), KUU_adj.clone(), VUU_adj.clone(), Vid.clone(), preference.clone(), Vpreference.clone(), structure.clone(), criterion1, criterion2)

            VGraph = Graph_pkg(Vgraph.clone(), Vedge.clone(), Vvalue.clone().type(torch.float).clone(), Vid.clone(), device=device)
            KGraph = Graph_pkg(Kgraph.clone(), Kedge.clone(), Kvalue.clone().type(torch.float).clone(), Kid.clone(), device=device)
            modelF4, model4, loss4 = inference_train(optimizer4, modelF4, model4, max_group_member, VGraph, KGraph, Vitem.clone(), Kitem.clone(), KUU_adj.clone(), VUU_adj.clone(), Vid.clone(), preference.clone(), Vpreference.clone(), structure.clone(), criterion1, criterion2)

            Batch_train_error[0].update(loss0, 1)
            Batch_train_error[1].update(loss1, 1)
            Batch_train_error[2].update(loss2, 1)
            Batch_train_error[3].update(loss3, 1)
            Batch_train_error[4].update(loss4, 1)

            print('Train {:d}/{:d}\t'.format(epoch, bth))
            
            if bth%5000==4999:
                best = test(epoch, bth, best, test_loader, modelF0, model0, modelF1, model1, modelF2, model2, modelF3, model3, modelF4, model4, max_group_member, device, n_classes, criterion_detail1, criterion_detail2)
                save_latest(modelF0, model0, 0)
                save_latest(modelF1, model1, 1)
                save_latest(modelF2, model2, 2)
                save_latest(modelF3, model3, 3)
                save_latest(modelF4, model4, 4)
            
        save_latest(modelF0, model0, 0)
        save_latest(modelF1, model1, 1)
        save_latest(modelF2, model2, 2)
        save_latest(modelF3, model3, 3)
        save_latest(modelF4, model4, 4)
        for i in range(5):
            with open('./record{}.txt'.format(i), 'a+') as f:
                f.write('Epoch:{}\tVal:{:.6f}\n'.format(epoch, Batch_train_error[i].avg))
        

def test(epoch, bth, best, test_loader, modelF0, model0, modelF1, model1, modelF2, model2, modelF3, model3, modelF4, model4, max_group_member, device, n_classes, criterion1, criterion2):
    Batch_test_error = [AverageMeter() for i in range(5)]
    Batch_test_error_RecE = [AverageMeter() for i in range(5)]
    Batch_test_error_RepE = [AverageMeter() for i in range(5)]
    Batch_test_error_SubE = [AverageMeter() for i in range(5)]
    Batch_test_error_RecR = [AverageMeter() for i in range(5)]
    Batch_test_error_RepR = [AverageMeter() for i in range(5)]
    Batch_test_error_SubR = [AverageMeter() for i in range(5)]
    modelF0.eval()
    model0.eval()
    modelF1.eval()
    model1.eval()
    modelF2.eval()
    model2.eval()
    modelF3.eval()
    model3.eval()
    modelF4.eval()
    model4.eval()
    with torch.no_grad():
        for bth, data in enumerate(test_loader, 0):
            Vgraph, Vedge, Vvalue, Kgraph, Kedge, Kvalue, Vid, Kid, Vitem, Kitem, preference, Vpreference, structure, KUU_adj, VUU_adj = data
                
            Vgraph = Vgraph.squeeze(0).to(device=device)
            Vedge = Vedge.squeeze(0).to(device=device)
            Vvalue = Vvalue.squeeze(0).to(device=device)
            Vid = Vid.squeeze(0).to(device=device)
            Kgraph = Kgraph.squeeze(0).to(device=device)
            Kedge = Kedge.squeeze(0).to(device=device)
            Kvalue = Kvalue.squeeze(0).to(device=device)
            Kid = Kid.squeeze(0).to(device=device)
            Vitem = Vitem.squeeze(0).to(device=device)
            Kitem = Kitem.squeeze(0).to(device=device)
            preference = preference.squeeze(0).to(device=device)
            structure = structure.squeeze(0).to(device=device)
            Vpreference = Vpreference.squeeze(0).to(device=device)
            KUU_adj = KUU_adj.to(device=device)
            VUU_adj = VUU_adj.to(device=device)

            VGraph = Graph_pkg(Vgraph.clone(), Vedge.clone(), Vvalue.clone().type(torch.float).clone(), Vid.clone(), device=device)
            KGraph = Graph_pkg(Kgraph.clone(), Kedge.clone(), Kvalue.clone().type(torch.float).clone(), Kid.clone(), device=device)
            loss, Recommend, Replace, Substitute, Tprefer, Tsocial, Thaptic = inference_test(max_group_member, modelF0, model0, criterion1, criterion2, VGraph, KGraph, Vitem.clone(), Kitem.clone(), KUU_adj.clone(), VUU_adj.clone(), Vid.clone(), preference.clone(), Vpreference.clone(), structure.clone())
            Batch_test_error[0].update(loss, 1)
            Batch_test_error_RecE[0].update(Recommend, 1)
            Batch_test_error_RepE[0].update(Replace, 1)
            Batch_test_error_SubE[0].update(Substitute, 1)
            Batch_test_error_RecR[0].update(Tprefer, 1)
            Batch_test_error_RepR[0].update(Tsocial, 1)
            Batch_test_error_SubR[0].update(Thaptic, 1)

            VGraph = Graph_pkg(Vgraph.clone(), Vedge.clone(), Vvalue.clone().type(torch.float).clone(), Vid.clone(), device=device)
            KGraph = Graph_pkg(Kgraph.clone(), Kedge.clone(), Kvalue.clone().type(torch.float).clone(), Kid.clone(), device=device)
            loss, Recommend, Replace, Substitute, Tprefer, Tsocial, Thaptic = inference_test(max_group_member, modelF1, model1, criterion1, criterion2, VGraph, KGraph, Vitem.clone(), Kitem.clone(), KUU_adj.clone(), VUU_adj.clone(), Vid.clone(), preference.clone(), Vpreference.clone(), structure.clone())
            Batch_test_error[1].update(loss, 1)
            Batch_test_error_RecE[1].update(Recommend, 1)
            Batch_test_error_RepE[1].update(Replace, 1)
            Batch_test_error_SubE[1].update(Substitute, 1)
            Batch_test_error_RecR[1].update(Tprefer, 1)
            Batch_test_error_RepR[1].update(Tsocial, 1)
            Batch_test_error_SubR[1].update(Thaptic, 1)

            VGraph = Graph_pkg(Vgraph.clone(), Vedge.clone(), Vvalue.clone().type(torch.float).clone(), Vid.clone(), device=device)
            KGraph = Graph_pkg(Kgraph.clone(), Kedge.clone(), Kvalue.clone().type(torch.float).clone(), Kid.clone(), device=device)
            loss, Recommend, Replace, Substitute, Tprefer, Tsocial, Thaptic = inference_test(max_group_member, modelF2, model2, criterion1, criterion2, VGraph, KGraph, Vitem.clone(), Kitem.clone(), KUU_adj.clone(), VUU_adj.clone(), Vid.clone(), preference.clone(), Vpreference.clone(), structure.clone())
            Batch_test_error[2].update(loss, 1)
            Batch_test_error_RecE[2].update(Recommend, 1)
            Batch_test_error_RepE[2].update(Replace, 1)
            Batch_test_error_SubE[2].update(Substitute, 1)
            Batch_test_error_RecR[2].update(Tprefer, 1)
            Batch_test_error_RepR[2].update(Tsocial, 1)
            Batch_test_error_SubR[2].update(Thaptic, 1)

            VGraph = Graph_pkg(Vgraph.clone(), Vedge.clone(), Vvalue.clone().type(torch.float).clone(), Vid.clone(), device=device)
            KGraph = Graph_pkg(Kgraph.clone(), Kedge.clone(), Kvalue.clone().type(torch.float).clone(), Kid.clone(), device=device)
            loss, Recommend, Replace, Substitute, Tprefer, Tsocial, Thaptic = inference_test(max_group_member, modelF3, model3, criterion1, criterion2, VGraph, KGraph, Vitem.clone(), Kitem.clone(), KUU_adj.clone(), VUU_adj.clone(), Vid.clone(), preference.clone(), Vpreference.clone(), structure.clone())
            Batch_test_error[3].update(loss, 1)
            Batch_test_error_RecE[3].update(Recommend, 1)
            Batch_test_error_RepE[3].update(Replace, 1)
            Batch_test_error_SubE[3].update(Substitute, 1)
            Batch_test_error_RecR[3].update(Tprefer, 1)
            Batch_test_error_RepR[3].update(Tsocial, 1)
            Batch_test_error_SubR[3].update(Thaptic, 1)

            VGraph = Graph_pkg(Vgraph.clone(), Vedge.clone(), Vvalue.clone().type(torch.float).clone(), Vid.clone(), device=device)
            KGraph = Graph_pkg(Kgraph.clone(), Kedge.clone(), Kvalue.clone().type(torch.float).clone(), Kid.clone(), device=device)
            loss, Recommend, Replace, Substitute, Tprefer, Tsocial, Thaptic = inference_test(max_group_member, modelF4, model4, criterion1, criterion2, VGraph, KGraph, Vitem.clone(), Kitem.clone(), KUU_adj.clone(), VUU_adj.clone(), Vid.clone(), preference.clone(), Vpreference.clone(), structure.clone())
            Batch_test_error[4].update(loss, 1)
            Batch_test_error_RecE[4].update(Recommend, 1)
            Batch_test_error_RepE[4].update(Replace, 1)
            Batch_test_error_SubE[4].update(Substitute, 1)
            Batch_test_error_RecR[4].update(Tprefer, 1)
            Batch_test_error_RepR[4].update(Tsocial, 1)
            Batch_test_error_SubR[4].update(Thaptic, 1)
            if bth>500:
                break
            
            
            
    for i in range(5):
        test_loss = Batch_test_error[i].avg
        test_gainE = Batch_test_error_SubE[i].avg-Batch_test_error_RepE[i].avg
        test_preferE = Batch_test_error_RecE[i].avg

        test_gainR = Batch_test_error_RecR[i].avg
        test_socialR = Batch_test_error_RepR[i].avg
        test_hapticR = Batch_test_error_SubR[i].avg

        if test_gainR+test_socialR+test_hapticR>best[i]:
            best[i] = test_gainR+test_socialR+test_hapticR
            try:
                save_path0 = './save{}/Epoch_{:d}_Batch_{:d}_Val_{:.6f}.pth'.format(i, epoch, bth, best[i])
                save_path1 = './save{}/Epoch_{:d}_Batch_{:d}_Val_{:.6f}F.pth'.format(i, epoch, bth, best[i])
                if i==0:
                    torch.save(model0.state_dict(), save_path0)
                    torch.save(modelF0.state_dict(), save_path1)
                elif i==1:
                    torch.save(model1.state_dict(), save_path0)
                    torch.save(modelF1.state_dict(), save_path1)
                elif i==2:
                    torch.save(model2.state_dict(), save_path0)
                    torch.save(modelF2.state_dict(), save_path1)
                elif i==3:
                    torch.save(model3.state_dict(), save_path0)
                    torch.save(modelF3.state_dict(), save_path1)
                elif i==4:
                    torch.save(model4.state_dict(), save_path0)
                    torch.save(modelF4.state_dict(), save_path1)
            except Exception as e:
                print(e)
        with open('./record{}.txt'.format(i), 'a+') as f:
            f.write('Test {:d}/{:d}\tloss:{:.3f}\tgainE:{:.3f}\tpreferE:{:.3f}\tgainR:{:.3f}\tsocialR:{:.3f}\thapticR:{:.3f}\n'.format(epoch, bth, test_loss, test_gainE, test_preferE, test_gainR, test_socialR, test_hapticR))
    return best

def main():
    device = 'cuda:2'
    batchsize = 1
    n_classes = 400
    max_group_member = 5
    max_scene_item = 10

    test_dataset = TestDataset('NewAmazon', split_low=0.2, split_high=1.1)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=int(8))

    train_dataset = TrainDataset('NewAmazon', split_low=0, split_high=0.8)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=int(8))
    
    modelF0 = Yolo_GCN(n_classes=n_classes, inference=False, n_users=len(train_dataset.Umap), n_items=len(train_dataset.Imap), max_group_member=max_group_member, max_scene_item=max_scene_item)
    modelF0 = modelF0.to(device=device)
    model0 = Group_Recommendation(n_classes=n_classes, max_group_member=max_group_member, max_scene_item=max_scene_item)
    model0 = model0.to(device=device)
    
    modelF1 = Yolo_GCN(n_classes=n_classes, inference=False, n_users=len(train_dataset.Umap), n_items=len(train_dataset.Imap), max_group_member=max_group_member, max_scene_item=max_scene_item)
    modelF1 = modelF1.to(device=device)
    model1 = Group_Recommendation(n_classes=n_classes, max_group_member=max_group_member, max_scene_item=max_scene_item)
    model1 = model1.to(device=device)
    
    modelF2 = Yolo_GCN(n_classes=n_classes, inference=False, n_users=len(train_dataset.Umap), n_items=len(train_dataset.Imap), max_group_member=max_group_member, max_scene_item=max_scene_item)
    modelF2 = modelF2.to(device=device)
    model2 = Group_Recommendation(n_classes=n_classes, max_group_member=max_group_member, max_scene_item=max_scene_item)
    model2 = model2.to(device=device)
    
    modelF3 = Yolo_GCN(n_classes=n_classes, inference=False, n_users=len(train_dataset.Umap), n_items=len(train_dataset.Imap), max_group_member=max_group_member, max_scene_item=max_scene_item)
    modelF3 = modelF3.to(device=device)
    model3 = Group_Recommendation(n_classes=n_classes, max_group_member=max_group_member, max_scene_item=max_scene_item)
    model3 = model3.to(device=device)
    
    modelF4 = Yolo_GCN(n_classes=n_classes, inference=False, n_users=len(train_dataset.Umap), n_items=len(train_dataset.Imap), max_group_member=max_group_member, max_scene_item=max_scene_item)
    modelF4 = modelF4.to(device=device)
    model4 = Group_Recommendation(n_classes=n_classes, max_group_member=max_group_member, max_scene_item=max_scene_item)
    model4 = model4.to(device=device)

    train(train_loader, test_loader, modelF0, model0, modelF1, model1, modelF2, model2, modelF3, model3, modelF4, model4, max_group_member, device, n_classes)

if __name__=='__main__':
    main()
