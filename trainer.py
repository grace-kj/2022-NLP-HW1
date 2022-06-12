# -*- coding: utf-8 -*-

import random
from tqdm import tqdm
import numpy as np
import torch
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score

from datamodule import DataModule
from model import BeepModel
from tokenizer import BeepTokenizer


class Trainer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BeepTokenizer()
        self.datamodule = DataModule(self.config, self.tokenizer)
        self.model = BeepModel(self.config, self.tokenizer).to('cuda')

        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()
        self.test_dataloader = self.datamodule.test_dataloader()

        self.optimizer = Adam(self.model.parameters(), lr = self.config.lr, eps = self.config.eps, weight_decay = self.config.weight_decay)

    def train(self):
        for epoch in tqdm(range(self.config.num_epochs)):
            self.model.zero_grad()
            self.model.train(True)

            train_bias_loss = 0.0
            train_hate_loss = 0.0

            for batch in tqdm(self.train_dataloader):
                inputs = {
                    'feature' : batch['feature'].to('cuda'),
                    'bias' : batch['bias'].to('cuda'),
                    'hate' : batch['hate'].to('cuda')
                }

                bias_loss, hate_loss, _, _ = self.model(**inputs) 
                
                train_bias_loss += bias_loss.item()
                train_hate_loss += hate_loss.item()

                loss = bias_loss + hate_loss
                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()

            print("|{:^79}|".format(" Epoch / Total Epoch : {} / {} ".format(epoch, self.config.num_epochs)))
            print("|{:^79}|".format(" Train Bias Loss : {:.4f} | Train Hate Loss : {:.4f} ".format(train_bias_loss / len(self.train_dataloader), train_hate_loss / len(self.train_dataloader))))

            self.model.train(False)

            self.valid()

    def valid(self):
        self.model.eval()

        val_bias_loss = 0.0
        val_hate_loss = 0.0

        for batch in tqdm(self.val_dataloader):
            inputs = {
                'feature' : batch['feature'].to('cuda'),
                'bias' : batch['bias'].to('cuda'),
                'hate' : batch['hate'].to('cuda')
            }

            bias_loss, hate_loss, _, _ = self.model(**inputs)
            
            val_bias_loss += bias_loss.item()
            val_hate_loss += hate_loss.item()

        print("|{:^79}|".format(" Valid Bias Loss : {:.4f} | Valid Hate Loss : {:.4f} ".format(val_bias_loss / len(self.val_dataloader), val_hate_loss / len(self.val_dataloader))))

    def test(self):
        self.model.eval()

        test_bias_loss = 0.0
        test_hate_loss = 0.0
        test_bias_label = []
        test_bias_pred = []
        test_hate_label = []
        test_hate_pred = []

        for batch in tqdm(self.test_dataloader):
            inputs = {
                'feature' : batch['feature'].to('cuda'),
                'bias' : batch['bias'].to('cuda'),
                'hate' : batch['hate'].to('cuda')
            }

            bias_loss, hate_loss, bias_pred, hate_pred = self.model(**inputs)
            
            test_bias_loss += bias_loss.item()
            test_hate_loss += hate_loss.item()

            test_bias_label.append(batch['bias'].cpu().detach().numpy())
            test_bias_pred.append(bias_pred.cpu().detach().numpy())
            test_hate_label.append(batch['hate'].cpu().detach().numpy())
            test_hate_pred.append(hate_pred.cpu().detach().numpy())
        
        test_bias_label, test_bias_pred = np.concatenate(test_bias_label), np.concatenate(test_bias_pred)
        test_hate_label, test_hate_pred = np.concatenate(test_hate_label), np.concatenate(test_hate_pred)
        test_bias_acc = accuracy_score(test_bias_label, test_bias_pred)
        test_hate_acc = accuracy_score(test_hate_label, test_hate_pred)
        test_bias_f1 = f1_score(test_bias_label, test_bias_pred, average = 'macro')
        test_hate_f1 = f1_score(test_hate_label, test_hate_pred, average = 'macro')

        print("|{:^79}|".format(" Test Bias Loss : {:.4f} | Test Hate Loss : {:.4f} | Test Bias Accuracy : {:.4f} | Test Hate Accuracy : {:.4f} | Test Bias F1 : {:.4f} | Test Hate F1 : {:.4f} ".format(test_bias_loss / len(self.test_dataloader), test_hate_loss / len(self.test_dataloader), test_bias_acc, test_hate_acc, test_bias_f1, test_hate_f1)))

    def train_and_test(self):
        self.train()
        self.test()