import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
from time import time


class BeamSearchNode(object):
    def __init__(self, hidden, previousNode, tokid, prob, length):

        self.h = hidden
        self.prev = previousNode
        self.tokid = tokid
        self.logp = prob
        self.len = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.len - 1 + 1e-6) + alpha * reward

def beam_decode(B, context, decoder, layer_embed, layer_nn, max_len, beam_size=5, beam_width=10):

    """contex: B, 1, H
    beam_size: Final output token size.(require beam-sentences of a single sentence)
    beam_width: Considered probability of tokens.
    """
    
    # Layers from generation models
    embedding, nn2vocab = layer_embed, layer_nn
    
    # Properties
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sos = 2
    eos = 3
    pad = 0
    unk = 1
    beam_batch = torch.LongTensor().to(device)
    
    # decoding goes sentence by sentence
    for idx in range(B):
        decoder_hidden = context.unsqueeze(1)[idx, :, :].unsqueeze(1) # sentence's context: 1, 1, H
        input = torch.LongTensor([[sos]]).to(device)

        # Initialize - previous hidden/ None previous node/ SOS/ zero prob/ length 1
        endnodes = []
        node = BeamSearchNode(decoder_hidden, None, input, 0, 1)
        Q = PriorityQueue()
        
        Q.put((-node.eval(), time(), node)) # Frist node(sos)
        Qsize = 1

        while True: 

            score, _, n = Q.get()
            input = n.tokid
            hidden = n.h

            if (n.len > max_len+1): # Eos and Not the sos
                endnodes.append((score, time(), n))
                if len(endnodes) >= beam_size:
                    break
                else:
                    continue
                
            input = embedding(input)
            output, hidden = decoder(input, hidden)
            logits = nn2vocab(output)
            
            logps, indexes = torch.topk(logits, beam_width) # candidates
            nextnodes = []

            # Recordind the logps & tokens
            for k in range(beam_width):
                decoded_t = indexes[:, 0, k].view(1, -1)
                logp = logps[:, 0, k].item()
                # logp: probability
                node = BeamSearchNode(hidden, n, decoded_t, n.logp + logp, n.len + 1)
                score = -node.eval()
                if not isinstance(score, float):
                    print(score)
                nextnodes.append((score, time(), node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, _, nn = nextnodes[i]
                try:
                    Q.put((score, time(), nn))
                except:
                    print('Debugging')
                    print(nn)
                    print(score)
                    prev = nn.prev
                    print(-prev.eval())
                # increase qsize
            Qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [Q.get() for _ in range(beam_width)]

        # End of searching, back-tracking the nodes for real outpu
        utterances = torch.LongTensor().to(device)
        for score, _, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            # back trace
            while (n.prev != None):
                n = n.prev
                utterance.append(n.tokid)

            utterance = utterance[::-1][1:]# Reverse the sequence
            utterance = torch.stack(utterance).view(1, 1, -1) # 1, 1, L
            utterances = torch.cat((utterances, utterance), dim=1) # 1, K, L
        beam_batch = torch.cat((beam_batch, utterances), dim=0) # B, K, L

    return beam_batch
