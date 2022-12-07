# -*- encoding: utf-8 -*-
'''
@File    :   generator.py
@Time    :   2022/10/05 11:50:19
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import os
import time
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class Generator(object):

    def __init__(self, args, logger, device):

        self.args = args
        self.num_workers = args.num_workers
        self.bs = args.train_batch_size
        self.logger = logger
        self.device = device

        self.logger.info("Loading dataset ... ")
        start = time.time()
        self._load_dataset()
        end = time.time()
        self.logger.info("Dataset is loaded: consume %.3f s" % (end - start))

    
    def _load_dataset(self):
        '''Load train, validation, test dataset'''

        return NotImplementedError

    
    def make_dataloaders(self):

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        eval_dataloader = DataLoader(self.eval_dataset,
                                     sampler=SequentialSampler(self.eval_dataset),
                                     batch_size=100,
                                     num_workers=self.num_workers)
        test_dataloader = DataLoader(self.test_dataset,
                                     sampler=SequentialSampler(self.test_dataset),
                                     batch_size=100,
                                     num_workers=self.num_workers)

        return train_dataloader, eval_dataloader, test_dataloader
    


