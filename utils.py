import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy
from torch.autograd import Variable
from operator import itemgetter

def clones(layer, N):
    return nn.ModuleList([deepcopy(layer) for i in range(N)])

def KL(mean, logv, \
       step, k, x0, anneal_f='logistic'):
    '''
    k: parameter controlled weight
    step: time-moving t
    #KL divergence = -0.5 x (2 log(sig) - sig^2 - mu^2 + 1)
    #logv.exp() == sig.pow^2. logv = 2log(sig)
    '''
    kl_loss = -0.5 * torch.sum(-mean.pow(2) -logv.exp() + logv + 1)
    
    if anneal_f == 'logistic':
        kl_weight= float(1/(1+np.exp(-k*(step - x0))))
    elif anneal_f == 'linear':
        kl_weight =  min(1, step/x0)

    return kl_loss, kl_weight

def vocab(counter):
    counter = sorted(counter.items(), key=itemgetter(1), reverse=True)
    return counter


class PositionwiseFeedForward(nn.Module):
    
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))
        pe[:, 1::2] = torch.cos(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))#torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):   # x.shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
        return self.dropout(x)
    
def get_pad_mask(seq, pad_idx): # (B, L), Prevent some pad token to have any attn
    return seq == pad_idx

def gen_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask.to(device)

def generate_square_subsequent_mask(sz):
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)
