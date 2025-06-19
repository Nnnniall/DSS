from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class MemoryModule(nn.Module):
    def __init__(self, nums, dim):
        super().__init__()
        self.dim = dim
        self.nums = nums
        self.weight = nn.Parameter(torch.empty(nums, dim))
        self.bias = None
        self.sig = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data):
        b, t, d = data.size()
        attention = self.sig(torch.einsum('btd,kd->btk', data, self.weight) / (self.dim ** 0.5))
        temporal_att = torch.topk(attention, self.nums // 16 + 1, dim=-1)[0].mean(-1)
        augment = torch.einsum('btk,kd->btd', attention[0:b // 2], self.weight)
        read_query = torch.cat((data[0:b // 2], augment), dim=-1)
        mem_score = 1 - temporal_att

        return mem_score, read_query