from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
import sys
import time
import numpy as np
import logging
from torchsummary import summary  # noqa
from pathlib import Path

import utils
from utils import makedir
import argparse
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

#from MODEL.dstanet import DSTANet
from MODEL.dylan_net_v10 import Dylan_MT_Net
# from Dataloader.Shrec_dataset import load_shrec_data, Sdata_generator, SConfig
from Dataloader.skeleton_loader import SConfig, Sdata_generator, Hand_Dataset


def train( model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    # ls = utils.LabelSmoothing()
    for batch_idx, (data1, target) in enumerate(train_loader):
        data1, target = data1.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data1)
        loss = criterion(output, target)
        # loss = ls(output, target)
        # L1_reg = 0
        # for param in model.parameters():
        #     L1_reg += torch.sum(torch.abs(param))
        # loss += 0.000001 * L1_reg  # lambda=0.001
        train_loss += loss.detach().item()
        loss.backward()
        optimizer.step()
    return train_loss


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, (data1, target) in enumerate(test_loader):
            data1, target = data1.to(device), target.to(device)
            output = model(data1)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    return acc


def main(att_drop, l_drop, weight_decay):
    # utils.init_seed(1)
    # Training settings
    batch_size = 32
    mid_layer = 4
    patience = 5
    clc_num = 14
    lr = 0.001
    frame_size = 120
    # utils.init_seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Config = SConfig(frame_size)

    train_data_generator = Sdata_generator('aug4_train', clc_num) #aug
    test_data_generator = Sdata_generator('val', clc_num)

    best_acc = 0
    X_0,X_1,X_2, Y = train_data_generator(Config)
    X_0 = torch.from_numpy(X_0).type('torch.FloatTensor')
    X_1 = torch.from_numpy(X_1).type('torch.FloatTensor')
    X_2 = torch.from_numpy(X_2).type('torch.FloatTensor')
    Y = torch.from_numpy(Y).type('torch.LongTensor')

    X_0_t,X_1_t,X_2_t, Y_t = test_data_generator(Config)
    X_0_t = torch.from_numpy(X_0_t).type('torch.FloatTensor')
    X_1_t = torch.from_numpy(X_1_t).type('torch.FloatTensor')
    X_2_t = torch.from_numpy(X_2_t).type('torch.FloatTensor')
    Y_t = torch.from_numpy(Y_t).type('torch.LongTensor')

    # trainset = TensorDataset(X_0, X_1, X_2, Y)
    trainset = Hand_Dataset(X_0, Y, frame_size, use_data_aug=False)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # testset = TensorDataset(X_0_t, X_1_t, X_2_t, Y_t)
    testset = Hand_Dataset(X_0_t, Y_t, frame_size, use_data_aug=False)
    test_loader = DataLoader(
        testset, batch_size=1000)


    # Net = DSTANet(config=config)
    Net = Dylan_MT_Net(3, mid_layer,
                       clc_num, num_node=22, num_frame=120,
                       n_layers=2, attn_heads=6,
                       dropout=att_drop, l_dropout=l_drop)

    model = Net.to(device)

    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay, amsgrad=False) # 0.001

    # optimizer = SGD(model.parameters(),lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0)
    # step = [30, 90]
    # scheduler = MultiStepLR(optimizer, step, gamma=args.gamma)
    # scheduler = CosineAnnealingLR(optimizer, args.epochs + 1, eta_min=5e-6)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.1, patience=patience, cooldown=0, min_lr=5e-6, verbose=False) #5e-6

    # scheduler = utils.GradualWarmupScheduler(optimizer, total_epoch=args.warm_up_epoch,
    #                                          after_scheduler=lr_scheduler_pre)


    for epoch in range(1, 200):
        train_loss = train(model, device, train_loader,
                           optimizer, epoch, criterion)
        acc = test(model, device, test_loader, criterion)
        #scheduler.step(train_loss)
        scheduler.step(train_loss)

        if acc > best_acc:
            best_acc = acc
            if best_acc > 93:
                torch.save(model.state_dict(), F"./bay_result/acc{np.around(best_acc, 2)}_model.pt")
    return best_acc



if __name__ == '__main__':
    #batch_size, mid_layer, att_drop, l_drop, lr, weight_decay, patience, label_smoothing,

    rf_bo = BayesianOptimization(
        main,
        {'att_drop': (0, 0.5),
         'l_drop': (0, 0.5),
            'weight_decay': (0, 0.5)
         },
        random_state=1
    )

    rf_bo.probe(
        {'att_drop': 0.0005,
         'l_drop': 0.3,
         'weight_decay': 0.0005}
    )
    rf_bo.maximize(n_iter=50)


    #|  27       |  90.83    |  0.07392  |  0.2411   |  0.009399 |  0.000946 |
    # | 55 | 90.24 | 0.05343 | 0.2532 | 0.004668 | 0.001771 |


# 1layers
#|  28       |  90.12    |  0.09253  |  0.2116   |  0.003666 |  0.000624 |

# 30 Aug
# |   iter    |  target   | att_drop  |  l_drop   | weight... |
# -------------------------------------------------------------
# |  1        |  89.29    |  0.05     |  0.2      |  0.001    |
# |  2        |  87.98    |  0.003587 |  0.0724   |  0.06834  |
# |  3        |  76.67    |  0.209    |  0.2      |  0.09333  |
# |  4        |  73.1     |  0.2842   |  0.1253   |  0.09847  |
# |  5        |  78.45    |  0.1489   |  0.2828   |  0.07925  |
# |  6        |  78.57    |  0.1738   |  0.2086   |  0.05232  |
# |  7        |  88.33    |  0.0005   |  0.2044   |  0.1      |
# |  8        |  89.52    |  0.0005   |  0.3      |  0.0005   |
# |  9        |  87.02    |  0.04498  |  0.2854   |  0.00949  |
# |  10       |  89.05    |  0.00124  |  0.1529   |  0.005987 |
# |  11       |  83.45    |  0.06902  |  0.1749   |  0.0541   |
# |  12       |  88.45    |  0.0005   |  0.0005   |  0.0005   |
# |  13       |  88.81    |  0.0005   |  0.0005   |  0.1      |
# |  14       |  88.45    |  0.000823 |  0.2177   |  0.004167 |
# |  15       |  88.57    |  0.1219   |  0.0005   |  0.0005   |
# |  16       |  83.81    |  0.1014   |  0.0005   |  0.1      |
# |  17       |  89.52    |  0.002615 |  0.1528   |  0.001738 |
# |  18       |  87.86    |  0.0005   |  0.3      |  0.1      |
# |  19       |  82.26    |  0.09524  |  0.1823   |  0.04506  |
# |  20       |  90.24    |  0.06284  |  0.07905  |  0.0005   |
# |  21       |  89.64    |  0.0005   |  0.0777   |  0.0005   |
# |  22       |  76.31    |  0.2162   |  0.2081   |  0.08923  |
# |  23       |  86.67    |  0.3      |  0.0005   |  0.0005   |
#  |  24       |  86.07    |  0.07849  |  0.02196  |  0.05419  |
# |  25       |  85.6     |  0.04645  |  0.09211  |  0.02681  |
# |  26       |  88.69    |  0.1444   |  0.07413  |  0.0005   |
# |  27       |  88.45    |  0.2106   |  0.0005   |  0.0005   |
# |  28       |  89.05    |  0.05882  |  0.03138  |  0.0005   |
# |  29       |  85.71    |  0.3      |  0.3      |  0.0005   |
#   |  30       |  79.4     |  0.1567   |  0.2629   |  0.03551  |
# |  31       |  88.69    |  0.06185  |  0.03198  |  0.001024 |
# |  32       |  90.36    |  0.09157  |  0.1276   |  0.0005   |
# |  33       |  77.26    |  0.2324   |  0.0005   |  0.1      |
# |  34       |  87.98    |  0.0005   |  0.2647   |  0.05331  |
# |  35       |  89.29    |  0.05025  |  0.1445   |  0.0005   |
# |  36       |  90.0     |  0.0005   |  0.02326  |  0.04861  |
# |  37       |  88.93    |  0.1537   |  0.1305   |  0.0005   |
# |  38       |  76.07    |  0.2181   |  0.248    |  0.03339  |
# |  39       |  88.21    |  0.2224   |  0.07275  |  0.0005   |
# |  40       |  84.17    |  0.02955  |  0.2473   |  0.02454  |
# |  41       |  88.93    |  0.0005   |  0.04563  |  0.1      |
# |  42       |  89.4     |  0.01706  |  0.04296  |  0.0005   |
# |  43       |  87.26    |  0.002348 |  0.2989   |  0.09986  |


# |   iter    |  target   | att_drop  |  l_drop   | weight... |
# -------------------------------------------------------------
# |  1        |  89.88    |  0.0005   |  0.3      |  0.0005   |
# |  2        |  89.64    |  0.1254   |  0.2162   |  1.144e-0 |
# |  3        |  85.24    |  0.09105  |  0.04445  |  0.009234 |
# |  4        |  86.67    |  0.05628  |  0.104    |  0.03968  |
# |  5        |  70.48    |  0.1619   |  0.126    |  0.06852  |
# |  6        |  89.4     |  0.06173  |  0.2635   |  0.002739 |
# |  7        |  90.6     |  0.0005   |  0.2006   |  0.0      |
# |  8        |  82.5     |  0.1599   |  0.2363   |  0.01749  |
# |  9        |  74.29    |  0.179    |  0.07868  |  0.07467  |
# |  10       |  67.02    |  0.06864  |  0.0471   |  0.09473  |
# |  11       |  72.98    |  0.218    |  0.2187   |  0.01947  |
# |  12       |  79.52    |  0.1413   |  0.1273   |  0.04719  |
# |  13       |  91.07    |  0.05974  |  0.2641   |  0.001068 |
# |  14       |  65.0     |  0.1469   |  0.2524   |  0.09791  |
#  |  15       |  68.57    |  0.09708  |  0.2292   |  0.0857   |
# |  16       |  60.6     |  0.2566   |  0.2159   |  0.0912   |
# |  17       |  91.55    |  0.01762  |  0.2412   |  0.001336 |Bn 16
# |  18       |  90.0     |  0.06937  |  0.1618   |  0.0      |
#  |  19       |  91.55    |  0.05867  |  0.2678   |  0.000345 |
# |  20       |  89.52    |  0.0005   |  0.1235   |  0.0      |

#
# |   iter    |  target   | att_drop  |  l_drop   | weight... |
# -------------------------------------------------------------
# |  1        |  93.33    |  0.0005   |  0.3      |  0.0005   |
# |  2        |  90.48    |  0.2088   |  0.3603   |  3.431e-0 |
# |  3        |  90.48    |  0.1515   |  0.0738   |  0.0277   |
# |  4        |  92.14    |  0.09354  |  0.1731   |  0.119    |
# |  5        |  86.67    |  0.2696   |  0.2099   |  0.2056   |
# |  6        |  89.64    |  0.1026   |  0.4391   |  0.008216 |
# |  7        |  84.4     |  0.1882   |  0.395    |  0.2323   |
# |  8        |  83.1     |  0.2664   |  0.3938   |  0.05247  |
# |  9        |  86.79    |  0.2981   |  0.1309   |  0.224    |
# |  10       |  91.07    |  0.1141   |  0.07822  |  0.2842   |
# |  11       |  72.38    |  0.3633   |  0.3644   |  0.05841  |
# |  12       |  86.31    |  0.2354   |  0.212    |  0.1416   |
# |  13       |  57.26    |  0.4777   |  0.3322   |  0.1412   |
# |  14       |  79.88    |  0.2446   |  0.4207   |  0.2937   |
# |  15       |  93.93    |  0.0005   |  0.0005   |  0.0      |
# |  16       |  92.38    |  0.000539 |  0.3292   |  0.2857   |
# |  17       |  82.86    |  0.3387   |  0.2742   |  0.115    |
# |  18       |  92.98    |  0.07081  |  0.07012  |  0.09505  |
# |  19       |  87.98    |  0.1374   |  0.3823   |  0.02006  |
# |  20       |  93.33    |  0.0005   |  0.1478   |  0.0      |
# |  21       |  86.67    |  0.2993   |  0.05679  |  0.03503  |
# |  22       |  82.86    |  0.307    |  0.2789   |  0.1961   |
# |  23       |  93.21    |  0.007986 |  0.009497 |  0.01031  |
# |  24       |  90.12    |  0.001339 |  0.4966   |  0.1517   |
# |  25       |  93.21    |  0.0005   |  0.0005   |  0.3      |
# |  26       |  93.57    |  0.0005   |  0.158    |  0.2313   |
# |  27       |  61.55    |  0.5      |  0.0005   |  0.3      |
# |  28       |  91.67    |  0.1154   |  0.03286  |  0.2112   |
# |  29       |  77.74    |  0.0005   |  0.5      |  0.3      |
# |  30       |  92.62    |  0.0005   |  0.5      |  0.0      |
# |  31       |  72.74    |  0.4197   |  0.2095   |  0.08883  |
# |  32       |  92.86    |  0.0005   |  0.3      |  0.1495   |
# |  33       |  93.33    |  0.0005   |  0.0005   |  0.1662   |
# |  34       |  91.55    |  0.1131   |  0.03115  |  0.2092   |
# |  35       |  90.48    |  0.0856   |  0.2456   |  0.3      |
# |  36       |  93.45    |  0.0005   |  0.1435   |  0.1173   |
# |  37       |  90.12    |  0.2249   |  0.2188   |  0.000738 |
# |  38       |  93.21    |  0.05205  |  0.02905  |  0.1109   |
# |  39       |  90.24    |  0.2087   |  0.0005   |  0.3      |
# |  40       |  92.38    |  0.0005   |  0.4119   |  0.0687   |
# |  41       |  93.69    |  0.0005   |  0.2207   |  0.3      |
# |  42       |  88.69    |  0.2297   |  0.004825 |  0.1643   |
#  |  43       |  92.62    |  0.06788  |  0.2233   |  0.003901 |
# |  44       |  92.98    |  0.002534 |  0.2597   |  0.2206   |
# |  45       |  94.4     |  0.0005   |  0.0962   |  0.3      | BN 32
# |  46       |  88.57    |  0.1277   |  0.4228   |  0.0392   |
# |  47       |  87.5     |  0.1283   |  0.4264   |  0.03661  |
# |  48       |  88.93    |  0.09518  |  0.3673   |  0.1229   |
# |  49       |  63.1     |  0.4055   |  0.3737   |  0.2669   |
# |  50       |  80.0     |  0.5      |  0.0005   |  0.0      |
# |  51       |  92.14    |  0.03427  |  0.39     |  0.004719 |
# |  52       |  93.69    |  0.0005   |  0.2335   |  0.07111  |
# |  53       |  94.05    |  0.0005   |  0.07149  |  0.2098   |
# |  54       |  90.95    |  0.2042   |  0.000909 |  0.000362 |
# |  55       |  89.05    |  0.05458  |  0.4688   |  0.07497  |
# |  56       |  89.17    |  0.2023   |  0.1285   |  0.2989   |
# =============================================================


