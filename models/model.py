# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/12/07 22:13:18
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import torch.nn as nn


class Model(nn.Module):

    def __init__(self) -> None:
        
        super().__init__()

    
    def forward(self, x):

        raise NotImplementedError




