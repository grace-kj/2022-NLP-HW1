# -*- coding: utf-8 -*-

import json, os
from typing import List, Dict
from konlpy.tag import Komoran

class BeepTokenizer:
    def __init__(self):
        self.komoran = Komoran()

        train = self._read_data()
        self.vocab = self._mk_vocab(train)

        self.unk_token_id = self.vocab['[UNK]']
        self.pad_token_id = self.vocab['[PAD]']
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def _read_data(self):
        train_path = os.path.join('./data/train.json')
        with open(train_path, 'r') as f:
            train = json.load(f)
        return train

    def _mk_vocab(self, train: List[Dict]) -> Dict:
        '''
            Inputs
                train = [
                    {
                        'comment' : '이주연님 되게 이쁘시다',
                        'bias' : 'none',
                        'hate' : 'none'
                    }
                    .
                    .
                    .
                    {
                        'comment' : '이제 그만좀해라',
                        'bias' : 'none',
                        'hate' : 'offensive'
                    }
                ]

            Outputs
                vocab = {
                    '[UNK]' : 0,
                    '[PAD]' : 1,
                    '!' : 2
                    .
                    .
                    .
                }

            Guide
                1. vocab 저장 시 오름차순으로 정렬.
                (예시) 
                ['가', 'c', 'b', '8', '1', '라', '!']
                -> {'[UNK]' : 0, '[PAD]' : 1, '!' : 2, '1' : 3, '8' : 4, 'b' : 5, 'c' : 6, '가' : 7, '라' : 8}
                2.형태소 분석시 품사 태그는 무시.
                (예시)
                ['그녀/NNG', '그녀/NP']
                -> {'[UNK]' : 0, '[PAD]: 1, '그녀' : 2} 
        '''
        vocab = {
            '[UNK]' : 0,
            '[PAD]' : 1
        }

        ############################################## EDIT ################################################
        vocab_dict = list()

        for i in range(len(train)):
          tokenized = self.tokenize(train[i]['comment'])
          vocab_dict += tokenized

        vocab_dict = list(set(vocab_dict))
        vocab_dict.sort()

        for i in range(len(vocab_dict)):
          vocab[vocab_dict[i]] = i+2


        ############################################## EDIT ################################################
        return vocab

    def tokenize(self, input: str) -> List[str]:
        '''
            Inputs
                '나는 라면을 먹었다'
            Outputs
                ['나', '는', '라면', '을', '먹', '었', '다']
        '''
        tokens = None

        ############################################## EDIT ################################################
        tokens = self.komoran.morphs(input)









        ############################################## EDIT ################################################

        return tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        '''
            Inputs
                ['나', '는', '라면', '을', '먹', '었', '다']
            Outputs
                [2, 3, 4, 5, 6, 7, 8]

            Guide
                1. 구성된 vocab을 잘 활용할것.
                2. Train 데이터로만 vocab를 구성하기 때문에, Valid와 Test 데이터의 token이 vocab에 존재하지 않을 수 있음.
                3. 이 경우 해당 token을 [UNK](unknown) token으로 처리.
        '''
        ids = None
        ############################################## EDIT ################################################
        output = list()

        for i in range(len(tokens)):
          if tokens[i] in self.vocab.keys():
            index = self.vocab[tokens[i]]
          else:
            index = 0
          output.append(index)


        ids = output

        ############################################## EDIT ################################################
        return ids

if __name__ == '__main__':
    '''
        tokenizer.py 의 method의 self-check 를 해보려면 tokenizer.py 파일을 실행시켜보세요.
        self-check 결과가 100% 맞지 않을 수 있습니다.
    '''
    
    tokenizer = BeepTokenizer()

    if len(tokenizer) == 12467 and tokenizer.vocab['!'] == 2 and tokenizer.vocab['사과'] == 5723 and tokenizer.vocab['힙합'] == 12465:
        print("Your '_mk_vocab' method is probably correct.")
    else:
        print("Your '_mk_vocab' method is probably wrong.")

    tokens = tokenizer.tokenize("자연어처리는 너무 어려워~!")
    if tokens == ['자연어', '처리', '는', '너무', '어렵', '어', '~', '!']:
        print("Your 'tokenize' method is probably correct.")
    else:
        print("Your 'tokenize' method is probably wrong.")
    
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if ids == [0, 10686, 2852, 2654, 7533, 7494, 409, 2]:
        print("Your 'convert_tokens_to_ids' method is probably correct.")
    else:
        print("Your 'convert_tokens_to_ids' method is probably wrong.")

