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

        self.embedding = nn.Embedding(len(tokenizer), 3, padding_idx = tokenizer.pad_token_id)
        self.bias_linear = nn.Linear(3, 3)
        self.hate_linear = nn.Linear(3, 3)

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
        ############################################## EDIT ################################################

        embedded_feature = self.embedding(feature)
        # embedded_feature.shape = (batch_size, max_length, 3)

        bias_output = torch.sum(self.bias_linear(embedded_feature), dim = 1)
        # bias_output.shape = (batch_size, 3)
        bias_loss = F.cross_entropy(bias_output, bias)
        print(bias_output.size(), bias.size())
        bias_pred = torch.argmax(bias_output, dim = -1)
        # bias_loss.shape = (,), bias_pred.shape = (batch_size,)

        hate_output = torch.sum(self.hate_linear(embedded_feature), dim = 1)
        # hate_output.shape = (batch_size, 3)
        hate_loss = F.cross_entropy(hate_output, hate)
        hate_pred = torch.argmax(hate_output, dim = -1)
        # hate_loss.shape = (,), hate_pred.shape = (batch_size,)

        ############################################## EDIT ################################################
        assert bias_loss.shape == ()
        assert hate_loss.shape == ()
        assert bias_pred.shape == (feature.shape[0],)
        assert hate_pred.shape == (feature.shape[0],)

        return bias_loss, hate_loss, bias_pred, hate_pred