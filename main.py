from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
from test import test
import option
from tqdm import tqdm
from utils import Visualizer, k_means_clustering
from config import *
import numpy as np
import torch.nn.functional as F

viz = Visualizer(env='DSS', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Model(n_features=args.feature_size, batch_size=args.batch_size, mem_num=args.mem_num, mem_dim=args.mem_size)
    for name, value in model.named_parameters():
        print(name)
    model = model.to(device)

    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')

    optimizer = optim.Adam(model.parameters(), lr=config.lr[0], weight_decay=0.0005)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1

    auc, gt, pred, result = test(test_loader, model, args, viz, device)

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train(loadern_iter, loadera_iter, model, optimizer, args, viz, device)

        if step % 5 == 0 and step > 200:

            auc, gt, pred, result = test(test_loader, model, args, viz, device)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save(model.state_dict(), './models/' + args.model_name + '{}.pth'.format(step))
                save_best_record(test_info, os.path.join('./outputs/', '{}-step-AUC.txt'.format(step)))
    torch.save(model.state_dict(), './models/' + args.model_name + 'final.pth')
