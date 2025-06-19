import torch
from utils import norm
torch.set_default_tensor_type('torch.FloatTensor')


def bce_loss(nor_top, abn_top, label, lamda0):
    loss_bce = torch.nn.BCELoss()
    scores = torch.cat((nor_top, abn_top), dim=0)
    scores = scores.squeeze(-1)
    loss = loss_bce(scores, label)
    return lamda0 * loss


def focal_loss(nor_top, abn_top, label, lamda0):
    scores = torch.cat((nor_top, abn_top), dim=0)
    loss = -(1 - scores) * (label * torch.log(scores)) - scores * ((1 - label) * torch.log(1 - scores))

    return lamda0 * loss.mean()


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    loss = torch.sum((arr2 - arr) ** 2)
    return lamda1 * loss


def sparsity(arr, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss


def recon_loss(en_feats, de_feats, lamda3):
    loss_recon = torch.nn.MSELoss()
    loss = loss_recon(de_feats, en_feats)
    return lamda3 * loss


def train(nloader, aloader, model, optimizer, args, viz, device):
    with torch.set_grad_enabled(True):
        model.train()

        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)
        input = torch.cat((ninput, ainput), 0).to(device)
        label = torch.cat((nlabel, alabel), 0).to(device)

        nor_top_scores, abn_top_scores, abn_bot_scores_enc, abn_top_scores_enc, en_feat, de_feat, abn_scores = model(input, is_train=True)

        loss_lin = bce_loss(nor_top_scores, abn_top_scores, label, 1.0)
        loss_enc = focal_loss(abn_bot_scores_enc, abn_top_scores_enc, label, 1.0)
        loss_recon = recon_loss(en_feat, de_feat, 1.0)
        loss_sparse = sparsity(abn_scores, 8e-3)

        cost = loss_lin + loss_enc + loss_recon + loss_sparse

        viz.plot_lines('loss', cost.item())

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()