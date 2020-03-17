from models import SVAE
from utils import *
from tracker import Tracker
from dataset import seq_data
import argparse
import numpy as np
from Transformer import Transformer
from multiprocessing import cpu_count
from collections import OrderedDict
from torch.utils.data import DataLoader


def main(args):

    splits = ['train', 'valid'] +  (['dev'] if args.test else [])
    print(args)
    #Load dataset
    datasets = OrderedDict()
    for split in splits:
        datasets[split]=seq_data(
            data_dir=args.data_dir,
            split=split,
            mt=args.mt,
            create_data=args.create_data,
            max_src_len=args.max_src_length,
            max_tgt_len=args.max_tgt_length,
            min_occ=args.min_occ
            )
    print('Data OK')
    #Load model
    model = Transformer(
        vocab_size=datasets['train'].vocab_size,
        embed_dim=args.embedding_dimension,
        hidden_dim=args.hidden_dimension,
        dropout=args.dropout,
        N_en=args.n_layer,
        N_de=args.n_layer,
        max_src_len=args.max_src_length, #influence in inference stage
        max_tgt_len=args.max_tgt_length, 
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx
        )
    print('Model OK')

    if torch.cuda.is_available():
        model = model.cuda()
    device = model.device
    #Training phase with validation(earlystopping)
    tracker = Tracker(patience=10, verbose=True) #record training history & es function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    step = 0
    
    for epoch in range(args.epochs):
        for split in splits:
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.n_batch,
                shuffle=(split=='train'),
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
                )
            if split == 'train':
                model.train()
            else:
                model.eval()
                
            #Executing
            for i, data in enumerate(data_loader):
                src, srclen,  tgt, tgtlen = \
                     data['src'], data['srclen'], data['tgt'], data['tgtlen']

                #FP
                logits, generations = model(src, srclen, tgt, tgtlen, split)

                #FP for groundtruth
                #h_pred, h_tgt = model.forward_gt(generations, tgt, tgtlen)

                #LOSS(weighted)
                NLL = model.loss(logits, tgt.to(device), data['tgtlen'])
                #GLOBAL = model.global_loss(h_pred, h_tgt)
                GLOBAL = 0
         
                loss = (NLL + GLOBAL)/data['src'].size(0)
                #BP & OPTIM
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step+=1
                    
                #RECORD & RESULT(batch)
                if i % 50 == 0 or i+1 == len(data_loader):
                    #NLL.data = torch.cuda.FloatTensor([NLL.data])
                    #KL.data = torch.cuda.FloatTensor([KL.data])
                    print("{} Phase - Batch {}/{}, Loss: {}, NLL: {}, G: {}".format(
                        split.upper(), i, len(data_loader)-1, loss, NLL, GLOBAL))
                tracker._elbo(torch.Tensor([loss]))
                if split == 'valid':
                    tracker.record(tgt, generations, datasets['train'].i2w,
                                   datasets['train'].pad_idx, datasets['train'].eos_idx, datasets['train'].unk_idx, None)                    

            #SAVING & RESULT(epoch)
            if split == 'valid':
                tracker.dumps(epoch, args.dump_file) #dump the predicted text.
            else:
                tracker._save_checkpoint(epoch, args.model_file, model.state_dict()) #save the checkpooint
            print("{} Phase - Epoch {} , Mean ELBO: {}".format(split.upper(), epoch, torch.mean(tracker.elbo)))

            tracker._purge()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='msmarco_data')
    parser.add_argument('--mt', action='store_true')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_src_length', type=int, default=100)
    parser.add_argument('--max_tgt_length', type=int, default=20)
    parser.add_argument('--min_occ', type=int, default=3) 
    parser.add_argument('-ep', '--epochs', type=int, default=5)
    parser.add_argument('-nb', '--n_batch', type=int, default=32)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--fasttext', action='store_true', default=False)
    parser.add_argument('--attention', action='store_true', default=False)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-ed', '--embedding_dimension', type=int, default=300)
    parser.add_argument('-hd', '--hidden_dimension', type=int, default=256)
    parser.add_argument('-wdp', '--word_dropout', type=float, default=0.1)
    parser.add_argument('-dp', '--dropout', type=float, default=0.1)
    parser.add_argument('-nl', '--n_layer', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', type=int, default=1)
    parser.add_argument('-af', '--af', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)
    parser.add_argument('-df', '--dump_file', type=str, default='dumps')
    parser.add_argument('-mf', '--model_file', type=str, default='checkpoint')

    args = parser.parse_args()

    main(args)
