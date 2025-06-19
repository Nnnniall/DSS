import torch
import torch.nn as nn


class DSS(nn.Module):
    """
    DSS non-learnable setting
    hyperparameter O , additional training paramters X
    """

    def __init__(self, delta):
        super(DSS, self).__init__()
        self.relu = nn.ReLU()
        self.delta = delta

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        b, c = x.size()

        x = self.relu(x)

        """ 1: max extractor """
        x_max = self.global_max_pool(x)
        x_max = x_max.expand_as(x)

        """ 2: suppression controller"""
        control = self.delta

        """ 3: suppressor"""
        x = torch.min(x, x_max * control)

        return x