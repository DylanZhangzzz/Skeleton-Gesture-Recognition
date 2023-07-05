import os
import torch
import random
import numpy as np
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
from MODEL.dylan_net_v9 import Dylan_MT_Net
# from Dataloader.Shrec_dataset import load_shrec_data, Sdata_generator, SConfig
from Dataloader.skeleton_loader import SConfig, Sdata_generator, Hand_Dataset

def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    # ls = utils.LabelSmoothing()
    for batch_idx, (data1, target) in enumerate(tqdm(train_loader)):
        data1, target = data1.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data1)
        loss = criterion(output, target)
        # loss = ls(output, target)
        train_loss += loss.detach().item()
        loss.backward()
        optimizer.step()
        # if batch_idx % 15 == 0:
        #     msg = ('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLr:{}'.format(
        #         epoch, batch_idx * len(data1), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item(), optimizer.state_dict()['param_groups'][0]['lr']))
        #     print(msg)
    return train_loss


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, (data1, target) in enumerate(tqdm(test_loader)):
            data1, target = data1.to(device), target.to(device)
            output = model(data1)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    msg = ('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(msg)

    return acc



# 这里不固定 random 模块的随机种子，因为 random 模块后续要用于超参组合随机组合。
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(1) # 为了模型除超参外其他部分的复现

param_grid = {
    'patience': [5],
    'learning_rate': [0.001],
    'batch_size': [16, 20, 32, 40, 64, 80, 128],
    'weight_decay': [0.0005],
    'att_head': [6],
    'att_layers': [2],
    'Drop': [0.0005],
    'L_Drop': [0.3],
    'mid_layers': [1]
}

MAX_EVALS = 20

best_score = 0
best_hyperparams = {}
best_epoch = 0

for i in range(MAX_EVALS):
    random.seed(i)  # 设置随机种子，每次搜索设置不同的种子，若种子固定，那每次选取的超参都是一样的
    hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}

    patience = hyperparameters['patience']
    learning_rate = hyperparameters['learning_rate']
    batch_size = hyperparameters['batch_size']
    weight_decay = hyperparameters['weight_decay']
    att_head = hyperparameters['att_head']
    att_layers = hyperparameters['att_layers']
    Drop = hyperparameters['Drop']
    L_Drop = hyperparameters['L_Drop']
    mid_layers = hyperparameters['mid_layers']
    print('lit:', i, hyperparameters)

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_generator = Sdata_generator('train', 28)  # aug
    test_data_generator = Sdata_generator('val', 28)

    C = SConfig(120)
    # Train, Test = load_data()
    X_0, X_1, X_2, Y = train_data_generator(C)
    X_0 = torch.from_numpy(X_0).type('torch.FloatTensor')
    X_1 = torch.from_numpy(X_1).type('torch.FloatTensor')
    X_2 = torch.from_numpy(X_2).type('torch.FloatTensor')
    Y = torch.from_numpy(Y).type('torch.LongTensor')

    X_0_t, X_1_t, X_2_t, Y_t = test_data_generator(C)
    X_0_t = torch.from_numpy(X_0_t).type('torch.FloatTensor')
    X_1_t = torch.from_numpy(X_1_t).type('torch.FloatTensor')
    X_2_t = torch.from_numpy(X_2_t).type('torch.FloatTensor')
    Y_t = torch.from_numpy(Y_t).type('torch.LongTensor')

    # trainset = TensorDataset(X_0, X_1, X_2, Y)
    trainset = Hand_Dataset(X_0, Y, 120, use_data_aug=True)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # testset = TensorDataset(X_0_t, X_1_t, X_2_t, Y_t)
    testset = Hand_Dataset(X_0_t, Y_t, 120, use_data_aug=False)
    test_loader = DataLoader(
        testset, batch_size=1000)

    # Net = DSTANet(config=config)#
    Net = Dylan_MT_Net(16, 4, 28,  num_node=22, num_frame=120, n_layers=att_layers, attn_heads=att_head, dropout=Drop, l_dropout=L_Drop)
    model = Net.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)  # 0.001
    # optimizer = SGD(model.parameters(),lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    criterion = nn.CrossEntropyLoss()

    # scheduler = MultiStepLR(optimizer, step, gamma=args.gamma)
    # scheduler = CosineAnnealingLR(optimizer, lr_args.max_epoch + 1, eta_min=5e-6)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.1, patience=patience, cooldown=0, min_lr=0, verbose=True)  # 5e-6

    # scheduler = utils.GradualWarmupScheduler(optimizer, total_epoch=args.warm_up_epoch,
    #                                          after_scheduler=lr_scheduler_pre)

    for epoch in range(1, 199 + 1):
        print('Epoch:', epoch)
        train_loss = train(model, device, train_loader,
                           optimizer, epoch, criterion)
        acc = test(model, device, test_loader, criterion)
        scheduler.step(train_loss)

        if acc > best_score:
            best_hyperparams = hyperparameters
            best_score = acc
            best_epoch = epoch
            # 你还可以在这一步保存模型，以最终得到最优的模型，如
            torch.save(model.state_dict(), "best_model.pt")
    print('Best acc===', best_score, 'at epoch:', best_epoch, 'with:', best_hyperparams)


#Best acc=== 90.83333333333333 at epoch: 112 with: {'patience': 12, 'learning_rate': 0.001, 'batch_size': 16, 'weight_decay': 0.0005, 'att_head': 6, 'att_layers': 2, '
#Drop': 0.0005, 'L_Drop': 0.3, 'mid_layers': 4}


# Best acc=== 90.23809523809524 at epoch: 67 with: {'patience': 5, 'learning_rate': 0.001, 'batch_size': 32, 'weight_decay': 0.001, 'att_head': 6, 'att_layers': 2, '
# Drop': 0.005, 'L_Drop': 0.05, 'mid_layers': 1}

# Best acc=== 90.83333333333333 at epoch: 90 with: {'patience': 5, 'learning_rate': 0.001, 'batch_size': 40, 'weight_decay': 0.001, 'att_head': 6, 'att_layers': 2, 'Drop': 0.005, 'L_Drop': 0.2, 'mid_layers': 1}
#

# Best acc=== 90.95238095238095 at epoch: 105 with: {'patience': 5, 'learning_rate': 0.001, 'batch_size': 8, 'weight_decay': 0.0005, 'att_head': 6, 'att_layers': 2, 'Drop': 0.0005, 'L_Drop': 0.3, 'mid_layers': 1}



