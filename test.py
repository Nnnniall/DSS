import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
from dataset import Dataset
import option
from config import *
from utils import Visualizer
from torch.utils.data import DataLoader
from model import Model


def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)
        gt = np.load(args.gt)
        result = {}
        for i, inputs in enumerate(dataloader):
            input, name = inputs
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            logits = model(input, is_train=False)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))
            result[name] = logits

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save('outputs/fpr.npy', fpr)
        np.save('outputs/tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('outputs/precision.npy', precision)
        np.save('outputs/recall.npy', recall)

        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)

        if args.dataset == "xd":
            np.save(f"outputs/pr_auc.npy", pr_auc)
            print('AP: ' + str(pr_auc))
            return pr_auc, list(gt), pred, result
        else:
            print('AUC: ' + str(rec_auc))
            return rec_auc, list(gt), pred, result


if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    viz = Visualizer(env='Test', use_incoming_socket=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Model(n_features=args.feature_size, batch_size=args.batch_size, mem_num=args.mem_num, mem_dim=args.mem_size)

    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)

    save_root = '/home/xinghongjie/xuchen/Project/DSS'

    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_root + '/models/DSS-xd.pth').items()})
    model.to(device)

    result = test(test_loader, model, args, viz, device)