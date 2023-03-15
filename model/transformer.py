#https://github.com/SamLynnEvans/Transformer
import math, copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 500, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, norm=True, dropout=True):
        if norm and dropout:
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.ff(x2))
        elif norm:
            x2 = self.norm_1(x)
            x = x + self.attn(x2,x2,x2,mask)
            x2 = self.norm_2(x)
            x = x + self.ff(x2)
        elif dropout:
            x2 = x
            x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
            x2 = x
            x = x + self.dropout_2(self.ff(x2))
        else:
            x2 = x
            x = x + self.attn(x2,x2,x2,mask)
            x2 = x
            x = x + self.ff(x2)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask=None, trg_mask=None, norm=True, dropout=True):
        if norm and dropout:
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff(x2))
        elif norm:
            x2 = self.norm_1(x)
            x = x + self.attn_1(x2, x2, x2, trg_mask)
            x2 = self.norm_2(x)
            x = x + self.attn_2(x2, e_outputs, e_outputs, src_mask)
            x2 = self.norm_3(x)
            x = x + self.ff(x2)
        elif dropout:
            x2 = x
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = x
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
            x2 = x
            x = x + self.dropout_3(self.ff(x2))
        else:
            x2 = x
            x = x + self.attn_1(x2, x2, x2, trg_mask)
            x2 = x
            x = x + self.attn_2(x2, e_outputs, e_outputs, src_mask)
            x2 = x
            x = x + self.ff(x2)
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout, pe_mode=0): #vocab_size, 
        super().__init__()
        self.N = N
        self.pe_mode = pe_mode
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, x, mask=None, norm=True, dropout=True, encoder_mask=None):
        if self.pe_mode==0:
            x=x
        else:
            x = self.pe(x)
        if encoder_mask is not None:
            x = x[:,encoder_mask]

        for i in range(self.N):
            x = self.layers[i](x, mask, norm, dropout)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout, pe_mode=0): #vocab_size, 
        super().__init__()
        self.N = N
        self.pe_mode = pe_mode
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, x, e_outputs, src_mask=None, trg_mask=None, norm=True, dropout=True):
        if self.pe_mode==0:
            x=x
        else:
            x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask, norm, dropout)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, trg_vocab, d_model, N, heads, dropout, pe_mode=0):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads, dropout, pe_mode)
        self.decoder = Decoder(d_model, N, heads, dropout, pe_mode)
        self.out = nn.Linear(d_model, trg_vocab)

    def initialize():
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
        for p in self.out.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    def forward(self, src, trg, src_mask=None, trg_mask=None, norm=True, dropout=True, encoder_mask=None):
        e_outputs = self.encoder(src, src_mask, norm, dropout, encoder_mask)
        # print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask, norm, dropout)
        output = self.out(d_output)
        return output

class BriefDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, mask=None, norm=True, dropout=True):
        if norm and dropout:
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff(x2))
        elif norm:
            x2 = self.norm_2(x)
            x = x + self.attn_2(x2, e_outputs, e_outputs, mask)
            x2 = self.norm_3(x)
            x = x + self.ff(x2)
        elif dropout:
            x2 = x
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, mask))
            x2 = x
            x = x + self.dropout_3(self.ff(x2))
        else:
            x2 = x
            x = x + self.attn_2(x2, e_outputs, e_outputs, mask)
            x2 = x
            x = x + self.ff(x2)
        return x

class ConnectionLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.image_encoder = EncoderLayer(d_model, heads, dropout=0.1)
        self.graph_encoder = EncoderLayer(d_model, heads, dropout=0.1)

        self.image_decoder = BriefDecoderLayer(d_model, heads, dropout)
        self.graph_decoder = BriefDecoderLayer(d_model, heads, dropout)

    def forward(self, image, graph, image_mask=None, graph_mask=None, norm=True, dropout=True):
        e_image = self.image_encoder(image, image_mask, norm, dropout)
        e_graph = self.graph_encoder(graph, graph_mask, norm, dropout)

        d_image = self.image_decoder(e_image, e_graph, image_mask, norm, dropout)
        d_graph = self.graph_decoder(e_graph, e_image, graph_mask, norm, dropout)

        return d_image, d_graph

class Connection(nn.Module):
    def __init__(self, d_model, N, heads, dropout=0.1, pe_mode=0): #vocab_size, 
        super().__init__()
        self.N = N
        self.pe_mode = pe_mode
        self.pe_image = PositionalEncoder(d_model, dropout=dropout)
        self.pe_graph = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(ConnectionLayer(d_model, heads, dropout), N)
        # self.norm1 = Norm(d_model)
        # self.norm2 = Norm(d_model)

        self.image_out = nn.Linear(d_model, d_model)
        self.graph_out = nn.Linear(d_model, d_model)

    def forward(self, image, graph, image_mask=None, graph_mask=None, norm=True, dropout=True, encoder_mask=None):
        
        if self.pe_mode==0:
            image=image
            graph=graph
        else:
            image = self.pe_image(image)
            graph = self.pe_graph(graph)
        
        if encoder_mask is not None:
            image = image[:, encoder_mask, :]

        for i in range(self.N):
            image_t, graph_t = self.layers[i](image, graph, image_mask, graph_mask, norm, dropout)
            image = image + image_t
            graph = graph + graph_t
        
        image_out = self.image_out(image)
        graph_out = self.graph_out(graph)
        return image_out, graph_out