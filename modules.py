import torch
import torch.nn as nn
from utils import clones, PositionwiseFeedForward, PositionalEncoding
import torch.nn.functional as F
from torch.nn import LayerNorm, MultiheadAttention, Dropout
from copy import deepcopy

class Encoder(nn.Module):
    def __init__(self, layer_en, N=1, tf=False): #norm layer
        super(Encoder, self).__init__()
        self.layers = clones(layer_en, N) #Stack N layer
        self.tf = tf
        if tf:
            self._activate_tf(layer_en)
            
    def _activate_tf(self, layer):
        # Preparation before layer loop
        self.pos_enc = PositionalEncoding(layer.embed_dim, layer.max_len)
        self.norm = LayerNorm(layer.embed_dim)
        self.dropout = Dropout(0.1)

    def forward(self, src, hidden=None, pad_mask=None):

        if self.tf:
            src = self.dropout(self.pos_enc(src))
            for attn_layer in self.layers:
                src, attn = attn_layer(src, pad_mask=pad_mask)
            return src, attn
        
        else:
            for layer in self.layers:
                src, hidden = layer(src, hidden)
            return src, hidden 
        
class Decoder(nn.Module):
    def __init__(self, layer_de, N=1, tf=False):
        super(Decoder, self).__init__()
        self.layers = clones(layer_de, N) #Stack N layer
        self.tf = tf
        if tf:
            self._activate_tf(layer_de)
        
    def _activate_tf(self, layer):
        # Preparation before layer loop
        self.pos_enc = PositionalEncoding(layer.embed_dim, layer.max_len)
        self.norm = LayerNorm(layer.embed_dim)
        self.dropout = Dropout(0.1)

    def forward(self, tgt, src_out, hidden=None, pad_mask=None, cross_mask=None):
        if self.tf:
            tgt = self.dropout(self.pos_enc(tgt))
            for attn_layer in self.layers:
                tgt, attn = attn_layer(tgt, src_out, pad_mask=pad_mask, cross_mask=cross_mask)

            #print(torch.sum(attn, dim=-1)) will be 1
            return tgt, attn
        else:
            for layer in self.layers:
                tgt, hidden = layer(tgt, hidden)
            return tgt, hidden


class RNNLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim=6, parallel=1):
        super(RNNLayer, self).__init__()
        self.rnn = nn.GRU(embed_dim, hidden_dim, parallel, batch_first=True)
        
    def forward(self, src, src_hidden=None):
        output, hidden = self.rnn(src, src_hidden) #src is the embedding
        return output, hidden


class ATTNLayer(nn.Module):

    def __init__(self, embed_dim, n_head, hidden_dim, inner_dim, dropout, max_len, cross):
        super(ATTNLayer, self).__init__()
        
        # Meta data of mattn
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.max_len = max_len
        self.dropout = dropout
        self.cross = cross

        # Multihead & Positionwise
        self.self_mattn = MultiheadAttention(embed_dim, n_head, dropout)
        self.pos_ff = PositionwiseFeedForward(embed_dim, inner_dim, dropout)
        self.norm = LayerNorm(embed_dim)
        
        # Cross attention
        if cross:
            self.cross_mattn = MultiheadAttention(embed_dim, n_head, dropout)

    def forward(self, seq, memory=None, pad_mask=None, cross_mask=None):
        # For preventing pad token involve attention weight.
        (src_pad_mask, tgt_pad_mask) = pad_mask

        if self.cross:
            output, _ = self.self_mattn(seq, seq, seq, key_padding_mask=tgt_pad_mask, attn_mask=cross_mask)
            output = output + self.norm(output)
            
            output, weight = self.cross_mattn(seq, memory, memory, key_padding_mask=src_pad_mask)
            output = output + self.norm(output)

            output = self.pos_ff(output) 
        else:
            output, weight = self.self_mattn(seq, seq, seq, key_padding_mask=src_pad_mask)
            output = output + self.norm(output)

            output = self.pos_ff(output)
        
        # Encoding: self-attn-weight/ Decoding: Cross-attn-weight
        return output, weight
