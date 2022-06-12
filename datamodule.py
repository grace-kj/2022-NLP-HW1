# -*- coding: utf-8 -*-

import json, os
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from typing import List

class BeepDataset(Dataset):
    def __init__(self, config, tokenizer, mode = 'train'):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config

        data = self._read_data(mode)
        self.data = self._data_processing(data)

    def _read_data(self, mode):
        data_path = os.path.join('./data', mode + '.json')
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def _convert_to_feature(self, comment: str) -> List[int]:
        '''
            Inputs
                '나는 라면을 먹었다'
            Outputs
                [1, 1, 1, 2, 3, 4, 5, 6, 7, 8]

            Guide
                1. tokenizer.py에서 구현한 BeepTokenizer 활용.
                2. 원활한 학습을 위해, config에 지정한 길이에 맞춰 truncation or padding 진행.
                3. truncation / padding의 앞 or 뒤 수행 위치에 따라 성능차이가 있을 수 있음.
                4. 위의 예시는 입력 token의 길이가 7, max_length가 10일때 padding을 앞에서 수행한 것. 
        '''
        feature = None

        ############################################## EDIT ################################################
        tokenized_idx = self.tokenizer.convert_tokens_to_ids(comment)

        feature = list()

        feature = tokenized_idx[:self.config.max_length]
        for i in range(self.config.max_length - len(tokenized_idx)):
          feature.insert(i, 1)

        ############################################## EDIT ################################################

        assert len(feature) == self.config.max_length

        return feature

    def _data_processing(self, data):
        bias_label = {
            'none' : 0,
            'gender' : 1,
            'others' : 2
        }

        hate_label = {
            'none' : 0,
            'offensive' : 1,
            'hate'  :2
        }

        output = []

        for dp in data:
            temp = {}

            temp['feature'] = self._convert_to_feature(dp['comment'])
            temp['bias'] = bias_label[dp['bias']]
            temp['hate'] = hate_label[dp['hate']]

            output.append(temp)

        return output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        output = {
            'feature' : torch.tensor(self.data[i]['feature'], dtype = torch.long),
            'bias' : torch.tensor(self.data[i]['bias'], dtype = torch.long),
            'hate' : torch.tensor(self.data[i]['hate'], dtype = torch.long)
        }

        return output

class DataModule:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def train_dataloader(self):
        train_dataset = BeepDataset(self.config, self.tokenizer, mode = 'train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size = self.config.train_batch_size, sampler = train_sampler)

        return train_dataloader

    def val_dataloader(self):
        val_dataset = BeepDataset(self.config, self.tokenizer, mode = 'val')
        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, batch_size = self.config.val_batch_size, sampler = val_sampler)

        return val_dataloader

    def test_dataloader(self):
        test_dataset = BeepDataset(self.config, self.tokenizer, mode = 'test')
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size = self.config.test_batch_size, sampler = test_sampler)

        return test_dataloader
