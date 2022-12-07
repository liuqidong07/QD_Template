# -*- encoding: utf-8 -*-
'''
@File    :   data.py
@Time    :   2022/10/05 10:50:59
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from torch.utils.data import Dataset


class Dataset(Dataset):
    '''The dataset for medication recommendation'''

    def __init__(self):
        
        super().__init__()

    def __len__(self):

        return NotImplementedError

    def __getitem__(self, item):

        return NotImplementedError
        
