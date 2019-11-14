# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



def get_sinusoid_encoding_table( d_hid, padding_idx=None,n_position=None,vocob_size=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    def get_vec_for_word():
       return [ 1 / np.power(10000, hid_j / d_hid) for hid_j in range(d_hid)]
    if n_position is not None:
        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    else:
        sinusoid_table = np.array([get_vec_for_word() for pos_i in range(vocob_size)])
    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    return torch.FloatTensor(sinusoid_table)

def init_weight(p):
    if len(p.shape) > 1:
        torch.nn.init.xavier_uniform((p))
    else:
        stdv = 1. / math.sqrt(p.shape[0])
        torch.nn.init.uniform_(p, a=-stdv, b=stdv)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight =nn.Parameter(torch.FloatTensor(in_features, out_features)) # nn.Parameter(torch.randn(in_features, out_features,dtype=torch.float)) #torch.randn(5, 7, dtype=torch.double)  # nn.Parameter(torch.FloatTensor(in_features, out_features))
        init_weight(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            init_weight(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_real = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_imag = nn.Parameter(torch.FloatTensor(in_features, out_features))

        self.bias = bias
        if bias:
            self.bias_real = nn.Parameter(torch.FloatTensor(out_features))
            self.bias_imag = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        for p in [self.weight_real,self.weight_imag,self.bias_real,self.bias_imag]:
            init_weight(p)

    def forward(self, x):
        x_real, x_imag = x
        hidden_real = torch.matmul(x_real, self.weight_real) - torch.matmul(x_imag, self.weight_imag)
        hidden_imag = torch.matmul(x_real, self.weight_imag) + torch.matmul(x_imag, self.weight_real)

        if self.bias is not None:
            output_real = hidden_real + self.bias_real
            output_imag = hidden_imag + self.bias_imag
        return output_real, output_imag

class CGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True,activation = "relu"):
        super(CGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = ComplexLinear(in_features, out_features, bias=bias)
        if bias:
            self.bias_real = nn.Parameter(torch.FloatTensor(out_features))
            self.bias_imag = nn.Parameter(torch.FloatTensor(out_features))
        for p in [self.bias_real,self.bias_imag]:
            init_weight(p)
    def forward(self, text, adj):
        hidden_real, hidden_imag = self.linear(text)

        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output_real = torch.matmul(adj, hidden_real) / denom
        output_imag = torch.matmul(adj, hidden_imag) / denom
        if self.bias is not None:
            output_real = output_real + self.bias_real
            output_imag = output_real + self.bias_imag
        return F.relu(output_real),F.relu(output_imag)


class CGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(CGCN, self).__init__()
        self.opt = opt
        self.concat = opt.concat
        self.schema = opt.schema
        self.complex = opt.complex
        self.pe = opt.pe
        self.device = opt.device
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float),freeze=False) # ,freeze=False
        vocob_size = torch.tensor(embedding_matrix, dtype=torch.float).size()[0]
        embedding_dim = torch.tensor(embedding_matrix, dtype=torch.float).size()[1]
        if self.complex :
            if self.schema == 0  :
                weight = get_sinusoid_encoding_table(embedding_dim,vocob_size = vocob_size, padding_idx=0)
                self.position_enc = nn.Embedding.from_pretrained(weight)  # , freeze=True
                self.position_enc.weight.requires_grad = True
                # self.position_enc = torch.nn.Embedding(vocob_size, embedding_dim)
            elif self.schema == 1:
                self.position_enc = torch.nn.Embedding(vocob_size, 1)

            elif self.schema == 2:
                self.position_enc = nn.Parameter(torch.Tensor(embedding_dim))  # , freeze=True
            elif self.schema ==3 or self.schema ==4:
                self.position_enc = torch.nn.Embedding(vocob_size, embedding_dim)
        else: # in the real case (complex =False), we will consider w/t PE, PE and TPE cases
            if self.pe == 1: # PE setting training case
                self.position_enc =  torch.nn.Embedding(512,embedding_dim) #nn.Parameter(torch.Tensor(512,embedding_dim)) 512 max length
            elif self.pe==2: # TPE setting, non training
                weight = get_sinusoid_encoding_table(embedding_dim,  n_position=512)
                self.position_enc = nn.Embedding.from_pretrained(weight,freeze=True)

            else:
                print("PE shoud be either 1 or 2")

        if self.complex and not self.concat:
            self.gc1 = CGraphConvolution( opt.hidden_dim,  opt.hidden_dim)
            self.gc2 = CGraphConvolution( opt.hidden_dim, opt.hidden_dim)
            self.complex_fc = ComplexLinear( opt.hidden_dim, opt.polarities_dim)

        elif self.complex and self.concat:
            self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
            self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
            self.fc = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)
        else:
            self.gc1 = GraphConvolution(opt.hidden_dim,  opt.hidden_dim)
            self.gc2 = GraphConvolution(opt.hidden_dim,  opt.hidden_dim)
            self.fc = nn.Linear( opt.hidden_dim, opt.polarities_dim)


        self.softmax = nn.Softmax()
        self.text_embed_dropout = nn.Dropout(0.3)

        self.get_embedding = self.get_mixed_Embedding if self.complex else self.get_real_Embedding


    def forward(self, inputs):
        text_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        # aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        # left_len = torch.sum(left_indices != 0, dim=-1)
        # aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        embed = self.get_embedding(text_indices ) #(enc_output_real + enc_output_phase) / 2

        x_1 =self.gc1(embed, adj)
        x_2 = self.gc2(x_1, adj)
        # print(x_2.shape)
        if self.complex and not self.concat:

            output = self.complex_fc([x_2[0].sum(1, keepdim=False),x_2[1].sum(1, keepdim=False)])
            norms = (torch.sqrt(torch.mul(output[0], output[0]) + torch.mul(output[1], output[1]))) / 1.5
            output = self.softmax(norms)

        else:
            output = self.fc(x_2.sum(1, keepdim=False))
            # print(output.shape)
            output = self.softmax(output)

        return output


    def get_mixed_Embedding(self,text_indices):
        text_lens = torch.sum(text_indices != 0, dim=-1)
        enc_output_real = self.embed(text_indices)

        embedding_dim = enc_output_real.size()[-1]
        batch_size = enc_output_real.size()[0]
        text_len = enc_output_real.size()[1]

        if self.schema == 0 or self.schema ==3 or self.schema ==4  :
            emb_peroid = self.position_enc(text_indices)
        elif self.schema == 1:  # vocab_size
            emb_peroid = self.position_enc(text_indices).repeat([1, 1,embedding_dim ])

        elif self.schema == 2:
            dimension_multiplier = torch.unsqueeze(self.position_enc, -2)
            dimension_multiplier = torch.unsqueeze(dimension_multiplier, -2)
            emb_peroid = dimension_multiplier.repeat([batch_size, text_len, 1])

        if self.schema ==4:
            enc_output_phase = emb_peroid
        else:
            pos_seq = torch.arange(1,text_lens.max() + 1, 1.0).to(self.device)
            pos_seq = torch.unsqueeze(pos_seq, -1)
            if self.schema != 3:  # vanilla complex setting without
                enc_output_phase = torch.mul(pos_seq, emb_peroid)
            else:
                enc_output_phase = emb_peroid
            enc_output = self.text_embed_dropout(enc_output_real)
            enc_output_phase = self.text_embed_dropout(enc_output_phase)
            cos = torch.cos(enc_output_phase)
            sin = torch.sin(enc_output_phase)

            enc_output_real = enc_output * cos
            enc_output_phase = enc_output * sin

        if self.concat:
            return torch.cat([enc_output_real,enc_output_phase],-1)
        else:
            return [enc_output_real,enc_output_phase]

    def get_real_Embedding(self,text_indices):
        assert self.complex == False, "wrong for real valued setting with complex being true"
        if self.pe==0:

            return self.embed(text_indices)
        else: # both for PE and TPE
            batch_size = text_indices.size()[0]
            text_lens = torch.sum(text_indices != 0, dim=-1)
            pos_seq = torch.LongTensor(torch.arange(1,text_lens.max() + 1, 1)).to(self.device)
            pos_seq = pos_seq.repeat([batch_size, 1])

            return self.embed(text_indices) + self.position_enc(pos_seq)



