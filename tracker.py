import numpy as np
import os
import json
import torch

class Tracker():
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.counter = 0
        self.best_score = -np.Inf
        self.es_flag = False
        self.val_loss_min = np.inf
        # initialize all the records
        self._purge()

    def _elbo(self, train_loss):
        loss = torch.cuda.FloatTensor([train_loss]) if torch.cuda.is_available() else torch.Tensor([train_loss]) 
        self.elbo = torch.cat((self.elbo, loss))

    def record(self, truth, predict, i2w, pad_idx, eos_idx, unk_idx, latent):
        '''note the predicted and groungtruth'''
        #predict = torch.argmax(predict, dim=2)
        gt, hat = [], []
        useless = [torch.tensor(pad_idx), torch.tensor(eos_idx), torch.tensor(unk_idx)]
        for i, (s1, s2) in enumerate(zip(truth.long(), predict.long())):
            gt.append(" ".join([i2w[str(int(idx))] for idx in s1 if idx not in useless])+"\n")
            hat.append(" ".join([i2w[str(int(idx))] for idx in s2 if idx not in useless])+"\n")

        self.gt += gt
        self.hat += hat
        self.z = torch.cat((self.z, latent))
        
    def dumps(self, epoch, file):
        if not os.path.exists(file):
            os.makedirs(file)
        with open(os.path.join(file+'/Target_E%i.txt'%epoch), 'w') as dump_tgt:
            dump_tgt.writelines(self.gt)
        with open(os.path.join(file+'/Predict_E%i.txt'%epoch), 'w') as dump_predict:
            dump_predict.writelines(self.hat)
            
    def _save_checkpoint(self, epoch, file, model):
        if not os.path.exists(file):
            os.makedirs(file)
        torch.save(model, os.path.join(file+'/Model_E%i.txt'%epoch))
        print('Model_E%i saved'%epoch)
            
    def _purge(self):
        '''
        elbo: (list) of loss on each pahse.
        gt: (list) of groundtruth on validation phase.
        hat: (list) of predicted text output
        z: (tensor) stacked output from latent space.
        '''
        self.gt, self.hat = [], []
        self.elbo = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.Tensor()
        self.z =  torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.Tensor()
        
    def es(self, val_loss, model, model_name):
        pass

