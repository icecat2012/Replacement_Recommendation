import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

class ConnectionG2C(nn.Module):
    def __init__(self, img_channel):
        super().__init__()
        self.img_channel = img_channel
        self.q_project = nn.Conv1d(img_channel, img_channel, 1, stride=1)
        self.k_project = nn.Conv1d(32, img_channel, 1, stride=1)
        self.v_project = nn.Conv1d(32, img_channel, 1, stride=1)
        self.m = nn.Softmax(dim=1)
        self.conv1 = Conv_Bn_Activation(img_channel, img_channel, 1, 1, 'leaky')
        self.conv2 = nn.Conv2d(img_channel, img_channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(img_channel, img_channel, 1, 1, 0)

    def forward(self, input_graph, input_image):
        BCWH = input_image.shape
        img = input_image.clone()
        input_graph = input_graph.permute(0,2,1) # B, D, N
        K = self.k_project(input_graph) # B, D', N
        V = self.v_project(input_graph) # B, D', N
        input_image = input_image.reshape((BCWH[0], BCWH[1], BCWH[2]*BCWH[3])) # B, C, WH
        Q = self.q_project(input_image) # B, D', WH

        Q = Q.permute(0,2,1) # B, WH, D'
        att = torch.bmm(Q, K) # B, WH, N
        att = self.m(att/np.sqrt(float(self.img_channel)))
        att = att.permute(0,2,1) # B, N, WH
        message = torch.bmm(V, att) # B, D', WH

        message = message.reshape((BCWH[0], BCWH[1], BCWH[2], BCWH[3])) # B, D', W, H
        message = self.conv2(self.conv1(message))
        return img + self.conv3(message)

class ConnectionC2G(nn.Module):
    def __init__(self, img_channel, graph_channel):
        super().__init__()
        self.img_channel = img_channel
        self.graph_channel = graph_channel
        self.q_project = nn.Conv1d(32, graph_channel, 1, stride=1)
        self.k_project = nn.Conv1d(img_channel, graph_channel, 1, stride=1)
        self.v_project = nn.Conv1d(img_channel, graph_channel, 1, stride=1)
        self.m = nn.Softmax(dim=1)
        self.conv1 = nn.Conv1d(graph_channel, graph_channel, 1, 1, 0)

    def forward(self, input_graph, input_image):
        BCWH = input_image.shape
        grahp = input_graph.clone()
        input_image = input_image.reshape((BCWH[0], BCWH[1], BCWH[2]*BCWH[3])) # B, C, WH
        K = self.k_project(input_image) # B, D', WH
        V = self.v_project(input_image) # B, D', WH
        input_graph = input_graph.permute(0,2,1) # B, D, N
        Q = self.q_project(input_graph) # B, D', N

        Q = Q.permute(0,2,1) # B, N, D'
        att = torch.bmm(Q, K) # B, N, WH
        att = self.m(att/np.sqrt(float(self.graph_channel)))
        att = att.permute(0,2,1) # B, WH, N
        message = torch.bmm(V, att) # B, D', N

        grahp = grahp.permute(0,2,1) # B, D, N
        message = grahp + self.conv1(message)
        message = message.permute(0,2,1) # B, N, D'
        return message