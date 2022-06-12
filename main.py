# -*- coding: utf-8 -*-

import argparse, random, time
import numpy as np
import torch

from trainer import Trainer

def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

class Dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def main():
    start = time.time()
    ############################################## EDIT ################################################
    config = {
        'seed' : 23,
        'num_epochs' : 6,
        'train_batch_size' : 16,
        'val_batch_size' : 16,
        'test_batch_size' : 16,
        'max_length' : 100,
        'lr' : 1e-5,
        'eps' : 1e-8,
        'weight_decay' : 0.0
    }
    ############################################## EDIT ################################################

    config = Dotdict(config)
    set_seed(config)

    trainer = Trainer(config)
    trainer.train_and_test()

    print("Execution time : {:.4f} sec".format(time.time() - start))
    
if __name__ == '__main__':
    main()