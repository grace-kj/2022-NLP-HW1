# -*- coding: utf-8 -*-

import torch.nn.functional as F
import torch.nn as nn
import torch

class BeepModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        '''
            Guide
                torch.nn에서 사용 가능한 모델은
                Embedding, Linear, RNN, CNN 으로 제한합니다.
        '''
        ############################################## EDIT ################################################
        self.embedding = nn.Embedding(len(tokenizer), 1024, padding_idx = tokenizer.pad_token_id)

        self.bias_rnn = nn.RNN(input_size=1024, hidden_size=1024, num_layers=1, batch_first=True, bidirectional=True)
        self.hate_rnn = nn.RNN(input_size=1024, hidden_size=1024, num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(1024*2, 128)
        self.fc2 = nn.Linear(128,3)
        self.sigmoid = nn.Sigmoid()

        self.bias_linear = nn.Linear(1024, 3)
        self.hate_linear = nn.Linear(1024, 3)

        ############################################## EDIT ################################################
    
    def forward(self, feature, bias, hate):
        '''
            Inputs
                feature.shape = (batch_size, max_length)
            
            Outputs
                bias_loss.shape = (,)
                hate_loss.shape = (,)
                bias_pred.shape = (batch_size,)
                hate_pred.shape = (batch_size,)
        '''
        bias_loss, hate_loss, bias_pred, hate_pred = None, None, None, None

        ############################################## EDIT ################################################
        
        embedded_feature = self.embedding(feature)
        
        # embedded_feature.shape = (batch_size, max_length, 3)
        #print("embedding size : ", embedded_feature.size())
        bias_output, h_n = self.bias_rnn(embedded_feature)

        #print("final output size : ", bias_output.size())
        dense_bias = self.fc1(bias_output[:,-1,:])
        #print("dense input size : ", bias_output[:,-1,:].size())
        dense_bias = self.fc2(dense_bias)
        outputs_bias=self.sigmoid(dense_bias)

        #print("final_output size : ", outputs_bias.size(), "bias size : ", bias.size())
        bias_loss = F.cross_entropy(outputs_bias, bias)
        bias_pred = torch.argmax(outputs_bias, dim = -1)
        
        hate_output, h_n = self.hate_rnn(embedded_feature)
        dense_hate = self.fc1(hate_output[:,-1,:])
        dense_hate = self.fc2(dense_hate)
        outputs_hate = self.sigmoid(dense_hate)

        hate_loss = F.cross_entropy(outputs_hate, hate)        
        hate_pred = torch.argmax(outputs_hate, dim = -1)
        '''

        bias_output = torch.sum(self.bias_linear(embedded_feature), dim = 1)
        bias_loss = F.cross_entropy(bias_output, bias)
        bias_pred = torch.argmax(bias_output, dim = -1)

        hate_output = torch.sum(self.hate_linear(embedded_feature), dim = 1)
        hate_loss = F.cross_entropy(hate_output, hate)
        hate_pred = torch.argmax(hate_output, dim = -1)
        '''
        ############################################## EDIT ################################################

        assert bias_loss.shape == ()
        assert hate_loss.shape == ()
        assert bias_pred.shape == (feature.shape[0],)
        assert hate_pred.shape == (feature.shape[0],)

        return bias_loss, hate_loss, bias_pred, hate_pred