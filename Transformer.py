import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, MultiheadAttention, Dropout
from modules import Encoder, Decoder, ATTNLayer
from utils import gen_mask, get_pad_mask, clones, PositionalEncoding, generate_square_subsequent_mask

class Transformer(nn.Module):
    
    def __init__(self, embed_dim=300, hidden_dim=256, inner_dim=2048,
                 n_head=2, N_en=6, N_de=6, dropout=0.1,
                 vocab_size=5000, sos_idx=2, eos_idx=3, pad_idx=0, unk_idx=1,
                 max_src_len=100, max_tgt_len=20, args=False):
        
        super(Transformer, self).__init__()

        #===Test the GPU availability
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #--Token indexes & Properties
        self.sos, self.eos, self.pad, self.unk = sos_idx, eos_idx, pad_idx, unk_idx
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.scale = embed_dim ** 0.5

        #===Base model(attn, enc, dec, ff)
        max_len = max(max_src_len, max_tgt_len)
        attn_enc_layer = ATTNLayer(
            embed_dim, n_head, hidden_dim, inner_dim, dropout, max_len, False)
        attn_dec_layer = ATTNLayer(
            embed_dim, n_head, hidden_dim, inner_dim, dropout, max_len, True)


        #===Main Archetecture(enc, dec)
        self.encoder = Encoder(attn_enc_layer, N_en, True)
        self.decoder = Decoder(attn_dec_layer, N_de, True)
        
        #===Embedding setting(src, tgt)
        self.embed = nn.Embedding(vocab_size, embed_dim)

        #===Fianl FC(logit2vocab)
        self.final = nn.Linear(embed_dim, vocab_size)

        #===Loss
        self.NLL = nn.NLLLoss(reduction='sum')

    def loss(self, logit, target, length):

        flat_logit=logit.reshape(-1, logit.size(2))
        flat_target=target.reshape(-1)

        nll_loss=self.NLL(flat_logit, flat_target)
        return nll_loss

    def forward(self, src, src_len, tgt, tgt_len, split='train'):
        
        n_batch=src.size(0)

        #Creating mask
        src_pad_mask = get_pad_mask(src, self.pad).to(self.device) #(B, S)
        tgt_pad_mask = get_pad_mask(tgt, self.pad).to(self.device) #(B, T)
        pad_mask = (src_pad_mask, tgt_pad_mask)

        attn_mask = generate_square_subsequent_mask(self.max_tgt_len)

        src = self.embed(src.to(self.device))
        tgt = self.embed(tgt.to(self.device))

        #permutation
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        enc_output, attn = self.encoder(src, pad_mask=pad_mask)
        output, attn = self.decoder(tgt, enc_output, pad_mask=pad_mask, cross_mask=attn_mask)

        #permutation
        output = output.permute(1, 0, 2) * (self.scale ** 0.5)

        logits = F.log_softmax(self.final(output), dim=-1)
        generations = torch.argmax(logits, dim=-1)
        
        return logits, generations
