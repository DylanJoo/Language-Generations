import os
import io
import utils
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict, Counter, OrderedDict
from nltk.tokenize import TweetTokenizer

class seq_data(Dataset):

    def __init__(self, data_dir='./msmarco_data/', split='train',
                 mt=False, create_data=False, **kwargs):
        super().__init__()

        self.max_src_len = kwargs.get('max_src_len', 100)
        self.max_tgt_len = kwargs.get('max_tgt_len', 20)
        
        self.min_occ = kwargs.get('min_occ', 3)

        self.data_dir = data_dir
        self.split = split
        self.mt = mt

        self.vocabs = dict()

        self.src = list()
        self.tgt = list()
        self.length = list()
        
        #try load the vocab, if not create then load
        try:
            self.load_vocab()
        except:
            self._create_vocab()
            
        if create_data:#augmented(first run the process)
            if self.mt:
                self._create_mt() #Invovle inequal-length target/input
            else:
                self.max_len = self.max_src_len
                self._create_data() #create data if no target(generate the next of src)
                
        self.load_data()

    @property
    def pad_idx(self):
        return self.w2i['<pad>']
    @property
    def sos_idx(self):
        return self.w2i['<sos>']
    @property
    def eos_idx(self):
        return self.w2i['<eos>']
    @property
    def unk_idx(self):
        return self.w2i['<unk>']
    @property
    def vocab_size(self):
        return len(self.w2i)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        idx = str(idx)
        return {
            'src': np.asarray(self.data[idx]['src']),
            'tgt': np.asarray(self.data[idx]['tgt']),
            'srclen': self.data[idx]['srclen'],
            'tgtlen': self.data[idx]['tgtlen']
        }
    
    def load_data(self):
        with io.open(self.get_path('data-'+self.split+'.json'), 'r') as data_file:
            self.data = json.load(data_file)

    def _create_mt(self):
        '''Create for seq2seq model(in mt/query-passage scenario)
        **target sequence is specified.
        '''
        tokenizer = TweetTokenizer(preserve_case=False)
        data = defaultdict(dict)
        with open(self.get_path('src-'+self.split+'.txt'), 'r') as file1,\
             open(self.get_path('tgt-'+self.split+'.txt'), 'r') as file2:

            for i, (line1, line2) in enumerate(zip(file1, file2)): 
                input = tokenizer.tokenize(line1)
                input = input[:self.max_src_len-1]
                input = input + ['<eos>']

                target = tokenizer.tokenize(line2)
                target = target[:self.max_tgt_len-1]
                target = target + ['<eos>']

                src_length = len(input)
                tgt_length = len(target)

                input.extend(['<pad>'] * (self.max_src_len-src_length))
                target.extend(['<pad>'] * (self.max_tgt_len-tgt_length))

                #Word to index
                input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
                target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]
                
                data[i]['src'] = input
                data[i]['tgt'] = target
                data[i]['srclen'] = src_length
                data[i]['tgtlen'] = tgt_length

        with io.open(self.get_path('data-'+self.split+'.json'), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

    def _create_data(self):
        '''Create the generative model's data from source, then preprocesing.
        src: tok1, tok2, tok3, tok4, ...<eos> (<pad>)
        tgt: tok1, tok2, tok3, tok4, ...<eos> (<pad>)
        src2(seq for decoder): <sos> + seq2[:-1]
        '''
        tokenizer = TweetTokenizer(preserve_case=False)
        data = defaultdict(dict)
        
        #Load input sequence, augmented seqeunce if no target.
        with open(self.get_path('src-'+self.split+'.txt'), 'r') as file:
            for i, line in enumerate(file):
                
                input = ['<sos>'] + tokenizer.tokenize(line)
                input = input[:self.max_len]
                
                target = input[1:self.max_len]
                target = target + ['<eos>']
                length = len(input)
                assert length == len(target), 'Input, target need to be equal-length.'

                input.extend(['<pad>'] * (self.max_len-length))
                target.extend(['<pad>'] * (self.max_len-length))               

                #Word to index
                input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
                target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]                
                
                data[i]['src'] = input
                data[i]['tgt'] = target
                data[i]['srclen'] = length
                data[i]['tgtlen'] = length

        with io.open(self.get_path('data-'+self.split+'.json'), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))
            
    def _create_vocab(self):
        assert self.split == 'train', "Vocabulary can ONLY be build from trainset"

        tokenizer = TweetTokenizer(preserve_case=False)
        
        c = Counter()
        w2i = OrderedDict()
        i2w = OrderedDict()
        tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        
        for i, st in enumerate(tokens):
            i2w[i] = st
            w2i[st] = i
            
        with open(self.get_path('src-'+self.split+'.txt'), 'r') as file:
            for line in file:
                c.update(tokenizer.tokenize(line))
            
        if self.mt:
            with open(self.get_path('tgt-'+self.split+'.txt'), 'r') as file:
                for line in file:
                    c.update(tokenizer.tokenize(line))
                    
        #collection of the vocabulary and its counts
        vocab_counts = utils.vocab(c)
            
        for i, (word, counts) in enumerate(vocab_counts):
            if counts > self.min_occ and word not in tokens:
                i2w[len(w2i)] = word
                w2i[word] = len(w2i)
                    
        assert len(w2i) == len(i2w)
        vocab = dict(w2i=w2i, i2w=i2w)
        
        with io.open(self.get_path('vocab.json'), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))
            
        self.load_vocab()
        #so that two dicts are in included.
        
    def load_vocab(self):
        with open(self.get_path('vocab.json'), 'r') as vocab_file:
            vocab = json.load(vocab_file)
        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']
        
    def get_path(self, file):
        return os.path.join(self.data_dir, file)
