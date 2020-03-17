import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from modules import Encoder, Decoder, RNNLayer
from BeamSearch import beam_decode
import utils
import random

class SVAE(nn.Module):

    def __init__(self,
                 embed_dim=300, hidden_dim=256, latent_dim=16, 
                 teacher_forcing=0, dropout=0, n_direction=1, n_parallel=1, 
                 max_src_len=100, max_tgt_len=20,
                 vocab_size=5000, sos_idx=2, eos_idx=3, pad_idx=0, unk_idx=1,
                 k=0.0025, x0=2500, af='logistic', attn=False,
                 args=False):
        
        super().__init__()
        #===Argument parser activated
        if args :
            vocab_size = args.vocab_size
            embed_dim, hidden_dim, latent_dim = args.embedding_dimension, args.hidden_dimension, args.latent_dimension
            teacher_forcing, dropout, n_direction, n_parallel = args.teacher_forcing, args.dropout, args.n_direction, args.n_parallel
            max_src_len, max_tgt_len = args.max_src_length, args.max_tgt_length
            sos_idx, eos_idx, pad_idx, unk_idx = args.sos_idx, args.eos_idx, args.pad_idx, args.unk_idx
            k ,x0, af = args.k, args.x0, args.af
            attn = args.attention
            
        #===Test the GPU availability
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        #===Parameters
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.hidden_n = n_direction * n_parallel #bidirectional or parallel layers
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.dropout = nn.Dropout(p=dropout)
        self.teacher_forcing = teacher_forcing

        #==Variational==
        self.k = k
        self.x0 = x0
        self.af = af

        #==Attention Mechanism
        self.attn = attn

        #===Tokens Indices
        self.sos, self.eos, self.pad, self.unk = sos_idx, eos_idx, pad_idx, unk_idx

        #===Embedding
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed.to(self.device)

        #===Base layers in en/de
        gru_layer_en = RNNLayer(embed_dim, hidden_dim, n_parallel)
        gru_layer_de = RNNLayer(embed_dim, hidden_dim, n_parallel)
        
        #===Main Archetecture(enc, dec)
        self.encoder = Encoder(gru_layer_en, 1)
        self.decoder = Decoder(gru_layer_de, 1)

        #===VAE( latent z space then to hidden context)
        self.hidden2mean = nn.Linear(hidden_dim * self.hidden_n, latent_dim)
        self.hidden2logv = nn.Linear(hidden_dim * self.hidden_n, latent_dim)
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim * self.hidden_n)

        #===Output for generating
        self.outputs2vocab = nn.Linear(hidden_dim * n_direction, vocab_size)

        #===Loss function
        self.NLL = nn.NLLLoss(reduction='sum', ignore_index=self.pad)

    def load_prt(self, weight):
        self.embed = nn.Embedding.from_pretrained(weight)
        self.embed.to(self.device)

    def kl(self, dist, step):
           #mu, logv, step
        mu, logv, _ = dist
        kl_loss, kl_weight = utils.KL(mu, logv, step, self.k, self.x0, self.af)
        return kl_loss * kl_weight


    def loss(self, logit, target, length, 
             mu, logv, step, k, x0, af='logistic'):
        '''Arrange the loss of gans in this func'''
        logit = logit[:, :torch.max(length)]
        target = target[:, :torch.max(length)]
        flat_logit = logit.reshape(-1, logit.size(2))
        flat_target = target.reshape(-1)

        nll_loss = self.NLL(flat_logit, flat_target)
        kl_loss, kl_weight = utils.KL(mu, logv, step, k, x0, af)

        return nll_loss, kl_loss, kl_weight

    def global_loss(self, h_pred, h_tgt):

        #Firstly, squeeze the first dim
        h_pred = h_pred.squeeze(0)
        h_tgt = h_tgt.squeeze(0)

        n_batch = h_tgt.size(0)
        score = torch.matmul(h_pred, h_tgt.t()) # (B, H) %*% (H, B) = (B, B)
        score_jj = torch.diag(score).unsqueeze(1).repeat(1, n_batch) #(B, B)
        score_mat = F.relu(score - score_jj + 1)
        loss = torch.sum(score_mat) 

        return loss

    def forward_gt(self, pred, tgt, tgt_len):

        n_batch = tgt.size(0)

        #forward passing for groundtruth(target)
        tgt = tgt[:, :-1]
        tgt = torch.cat((torch.LongTensor([[self.sos]]*n_batch), tgt), dim=1).to(self.device)
        tgt = self.embed(tgt)
        tgt = pack(tgt, tgt_len, batch_first=True, enforce_sorted=False)
        #concat the <sos> token before the sentences, but remain the 

        #forward passing for generation(prediction)
        pred = pred[:, :-1]
        pred = torch.cat((torch.LongTensor([[self.sos]]*n_batch).to(self.device), pred), dim=1)
        pred = self.embed(pred)
        pred = pack(pred, tgt_len, batch_first=True, enforce_sorted=False)

        #encode each to obtain the hidden embediing
        _, h_pred = self.encoder(pred)
        _, h_tgt = self.encoder(tgt)

        return h_pred, h_tgt


    def forward(self, src, src_len, tgt, tgt_len, split='train', global_loss=False):
        
        src = src.to(self.device)
        assert src.size(0) == tgt.size(0), 'Batch size unequal...'
        n_batch = tgt.size(0)

        #Token2Embed
        src = self.embed(src)

        #Sequence2Pack
        src = pack(src, src_len, batch_first=True, enforce_sorted=False)
        
        #Encoding
        _, hidden = self.encoder(src)

        #--Bidirectional--
        if self.hidden_n != 1:
            hidden = hidden.view(batch_size, self.hidden_dim*self.hidden_n)
        else:
            hidden = hidden.squeeze()

        #Re-regularization
        z_ = torch.rand([n_batch, self.latent_dim]).to(self.device)
        mu = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv) #? why exponential
        z = z_ * std + mu

        #LatentSpace
        context = self.latent2hidden(z)  
        if self.hidden_n != 1:            # unflatten the hidden
            context = context.view(self.hidden_n, batch_size, self.hidden_dim)
        else:
            context = context.unsqueeze(0)

        # --Teacher forcing with probability 
        if (self.teacher_forcing > random.uniform(0,1)):
            teacher = True
        else:
            teacher = False

        ## --If validation, select NO teacher forcing
        if split == 'valid':
            teacher = False

        #Decoding, So far no worddropout
        t=0
        if teacher:
            tgt = tgt[:, :-1]
            tgt = torch.cat((torch.LongTensor([[self.sos]]*n_batch), tgt), dim=1).to(self.device)
            tgt = self.embed(tgt)
            tgt = pack(tgt, tgt_len, batch_first=True, enforce_sorted=False)
            pad_output, _ = self.decoder(tgt, context)
            output, output_len = unpack(pad_output, batch_first=True, total_length=self.max_tgt_len)
            logits = F.log_softmax(self.outputs2vocab(output), dim=-1)
            generations = torch.argmax(logits, dim=-1)
            
        else:
            #Initial with No groundtruth
            input = torch.LongTensor([[self.sos]]*n_batch).to(self.device)
            generations = torch.LongTensor(n_batch, 0).to(self.device)
            logits = []
            while(t<self.max_tgt_len):
                input = self.embed(input)
                output, context = self.decoder(input, context)

                logit = self.outputs2vocab(output).squeeze()  #(n_batch, vocab_size)
                
                """ Sampling for the next token.
                logit = F.Softmax(logit, dim=1)
                input = logit.multinomial(1)
                logit = torch.log(logit)
                """
                # Original Setting: logsoftmax with nllloss.
                logit = F.log_softmax(logit, dim=1)
                # MLE
                input = torch.argmax(logit, dim=1).unsqueeze(1)

                logits.append(logit)  
                generations = torch.cat((generations, input), dim=1)
                
                #if teacher_forcing:
                #    input = tgt[:, t].unsqueeze(1)
                t=t+1
                # In GAN, Output the predict generations
            logits = torch.stack(logits, dim=1)
                
        return logits, (mu, logv, z), generations
    
        # In SVAE
        #return logits, (mu, logv, z) #generations
    

    def inference(self, n=4, z=None):
        '''Infernece from 'assigned sentence' or 'random sentence'
        '''
        if z is None:
            n_batch = n
            z = torch.randn([n_batch, self.latent_dim]).to(self.device)
        else:
            z = z.to(self.device)
            n_batch = z.size(0)

        context = self.latent2hidden(z)
        
        if self.hidden_n != 1:            # unflatten the hidden
            context = context.view(self.hidden_n, n_batch, self.hidden_dim)
        else:
            context = context.unsqueeze(0)
        
        generations = torch.LongTensor(n_batch, self.max_tgt_len).fill_(self.pad)

        t = 0

        input = torch.LongTensor([[self.sos]]*n_batch).to(self.device)
        generations = torch.LongTensor(n_batch, 0).to(self.device)
        logits = []
        while(t<self.max_tgt_len):
            input = self.embed(input)
            output, _ = self.decoder(input, context)

            logit = self.outputs2vocab(output).squeeze()  #(n_batch, vocab_size)
                # Original Setting: logsoftmax with nllloss.
            logit = F.log_softmax(logit, dim=1)
                # MLE
            input = torch.argmax(logit, dim=1).unsqueeze(1)

            logits.append(logit)  
            generations = torch.cat((generations, input), dim=1)
                
                #if teacher_forcing:
                #    input = tgt[:, t].unsqueeze(1)
            t=t+1
                # In GAN, Output the predict generations
        logits = torch.stack(logits, dim=1)
                
        return generations, z
    
    def test(self, src, src_len):
        '''for the real scenario'''
        n_batch = src.size(0)
        src = src.to(self.device)
        #src==>embed==>rnn-packed
        src = self.embed(src)
        src = pack(src, src_len, batch_first=True, enforce_sorted=False)
        #===Encoding===
        _, hidden = self.encoder(src)
        if self.hidden_n != 1:             # flatten the hidden
            hidden = hidden.view(n_batch, self.hidden_dim*self.hidden_n)
        else:
            hidden = hidden.squeeze()

        #z_(before) reparameterize with mu and sigma (prior)
        z_ = torch.rand([n_batch, self.latent_dim]).to(self.device)
        mu = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)
        z = z_ * std + mu

        context = self.latent2hidden(z)

        #Beam searching
        generations = beam_decode(B=n_batch,
                context=context,
                decoder = self.decoder, 
                layer_embed=self.embed,
                layer_nn=self.outputs2vocab,
                max_len=self.max_tgt_len,
                beam_size=1, beam_width=3)

        generations = generations.view(-1, self.max_tgt_len)

        return generations, z
