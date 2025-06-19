import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
import torch.nn.functional as F
from torch.nn import init
import math
from restrain import DSS
from memory import MemoryModule
from einops import rearrange
import numpy as np

torch.set_default_tensor_type('torch.FloatTensor')


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Temporal(nn.Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 4, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(2 * inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, d = x.size()
        qkvt = self.to_qkv(x).chunk(4, dim=-1)
        q, k, v, t = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkvt)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn1 = self.attend(dots)

        tmp_ones = torch.ones(n).to(x.device)
        tmp_n = torch.linspace(1, n, n).to(x.device)
        tg_tmp = torch.abs(tmp_n * tmp_ones - tmp_n.view(-1, 1))
        attn2 = torch.exp(-tg_tmp / torch.exp(torch.tensor(1.)))
        attn2 = (attn2 / attn2.sum(-1)).unsqueeze(0).unsqueeze(1).repeat(b, self.heads, 1, 1)

        out = torch.cat([torch.matmul(attn1, v), torch.matmul(attn2, t)], dim=-1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Classifier(nn.Module):
    def __init__(self, input_feature_dim, dropout=0.7):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(input_feature_dim, 128), nn.ReLU(),
                                        nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x):
        logits = self.classifier(x)
        return logits


class Model(nn.Module):
    def __init__(self, n_features, batch_size, mem_num, mem_dim):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.embedding = Temporal(n_features, 512)
        self.attention = Transformer(512, 1, 1, 256, 512, dropout=0.5)
        self.cls_head = Classifier(512, dropout=0.7)
        self.decoder = nn.Linear(1024, n_features)
        self.memory = MemoryModule(mem_num, mem_dim)
        self.restrain = DSS(0.55)
        self.apply(weight_init)

    def forward(self, x, is_train):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        en_feat = x[0:b * n // 2]
        x = self.embedding(x)
        x = self.attention(x)
        lin_scores = self.cls_head(x)
        lin_scores = lin_scores.squeeze(-1)
        mem_scores, read_query = self.memory(x)
        mem_scores = mem_scores.squeeze(-1)
        de_feat = self.decoder(read_query)
        scores = (lin_scores + mem_scores) / 2

        if is_train:
            scores = scores.view(b, n, -1).mean(1)
            # nor_top
            nor_scores = scores[0:b // 2]
            nor_top_scores = torch.topk(nor_scores, t // 16 + 1, dim=-1)[0]
            nor_top_scores = torch.mean(nor_top_scores, dim=-1)
            # abn_top
            abn_scores = scores[b // 2:]
            abn_top_scores = torch.topk(abn_scores, t // 16 + 1, dim=-1)[0]
            abn_top_scores = torch.mean(abn_top_scores, dim=-1)

            abn_scores_sup = self.restrain(abn_scores)
            abn_scores_enc = torch.where(abn_scores > 0.85, abn_scores_sup, abn_scores)
            # abn_top
            abn_top_scores_enc = torch.topk(abn_scores_enc, t // 16 + 1, dim=-1)[0]
            abn_top_scores_enc = torch.mean(abn_top_scores_enc, dim=-1)
            # abn_bot
            abn_bot_scores_enc = torch.topk(abn_scores_enc, t // 16 + 1, dim=-1, largest=False)[0]
            abn_bot_scores_enc = torch.mean(abn_bot_scores_enc, dim=-1)

            return nor_top_scores, abn_top_scores, abn_bot_scores_enc, abn_top_scores_enc, en_feat, de_feat, abn_scores

        else:
            logits = scores.view(b, n, -1).mean(1)
            logits = logits.unsqueeze(dim=-1)

            return logits
