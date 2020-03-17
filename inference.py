import os
import json
import torch
import argparse
import numpy as np
from models import SVAE


def main(args):
    
    def interpolate(start, end, steps):
        interpolation = np.zeros((start.shape[0], steps + 2))
        for dim, (s,e) in enumerate(zip(start,end)):
            interpolation[dim] = np.linspace(s,e,steps+2)

        return interpolation.T

    def idx2word(sent_list, i2w, pad_idx):
        sent = []
        for s in sent_list:
            sent.append(" ".join([i2w[str(int(idx))] \
                                 for idx in s if int(idx) is not pad_idx]))
        return sent

    with open(args.data_dir+'/vocab.json', 'r') as file:
        vocab = json.load(file)
    w2i, i2w = vocab['w2i'], vocab['i2w']

    #Load model
    model = SVAE(
        vocab_size=len(w2i),
        embed_dim=args.embedding_dimension,
        hidden_dim=args.hidden_dimension,
        latent_dim=args.latent_dimension,

        teacher_forcing=False,
        dropout=args.dropout,
        n_direction= (2 if args.bidirectional else 1),
        n_parallel=args.n_layer,
        
        max_src_len=args.max_src_length, #influence in inference stage
        max_tgt_len=args.max_tgt_length,
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        )

    path = os.path.join('checkpoint', args.load_checkpoint)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    model.load_state_dict(torch.load(path))
    print("Model loaded from %s"%(path))

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    samples, z = model.inference(n=args.num_samples)
    print('----------SAMPLES----------')
    print(*idx2word(sent_list=samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    z1 = torch.randn([args.latent_dimension]).numpy()
    z2 = torch.randn([args.latent_dimension]).numpy()
    z = torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float()
    samples, _ = model.inference(z=z)
    print('-------INTERPOLATION-------')
    print(*idx2word(sent_list=samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str, default='Model_E2.txt')
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('--data_dir', type=str, default='msmarco_data')
    parser.add_argument('--max_src_length', type=int, default=100)
    parser.add_argument("--max_tgt_length", type=int, default=20)

    parser.add_argument('-ed', '--embedding_dimension', type=int, default=300)
    parser.add_argument('-hd', '--hidden_dimension', type=int, default=256)
    parser.add_argument('-ld', '--latent_dimension', type=int, default=16)
    parser.add_argument('-wdp', '--word_dropout', type=float, default=0.1)
    parser.add_argument('-dp', '--dropout', type=float, default=0.5)
    parser.add_argument('-nl', '--n_layer', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    args = parser.parse_args()

    assert 0 <= args.word_dropout <= 1

    main(args)
