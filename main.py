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
from MODEL.dylan_net_v12 import Dylan_MT_Net
# from Dataloader.Shrec_dataset import load_shrec_data, Sdata_generator, SConfig
from Dataloader.skeleton_loader import SConfig, Sdata_generator


def train(args, model, device, train_loader, optimizer, epoch, criterion, logging):
    model.train()
    train_loss = 0
    # ls = utils.LabelSmoothing()
    for batch_idx, (data1, data2, data3, target) in enumerate(tqdm(train_loader)):
        data1, data2, data3, target = data1.to(device), data2.to(device), data3.to(device), target.to(device)
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
        if batch_idx % args.log_interval == 0:
            msg = ('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLr:{}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), optimizer.state_dict()['param_groups'][0]['lr']))
            print(msg)
            logging.info(msg)

    history['train_loss'].append(train_loss)
    return train_loss


def test(model, device, test_loader, logging, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, (data1, data2, data3, target) in enumerate(tqdm(test_loader)):
            data1, data2, data3, target = data1.to(device), data2.to(device), data3.to(device), target.to(device)
            output = model(data1)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(correct / len(test_loader.dataset))
    acc = 100. * correct / len(test_loader.dataset)
    msg = ('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(msg)
    logging.info(msg)

    return acc

def test_val(model, device, test_loader, logging, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, (data1, data2, data3, target) in enumerate(tqdm(test_loader)):
            data1, data2, data3, target = data1.to(device), data2.to(device), data3.to(device), target.to(device)
            output = model(data1)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    history['val_loss'].append(test_loss)
    history['val_acc'].append(correct / len(test_loader.dataset))
    acc = 100. * correct / len(test_loader.dataset)
    msg = ('Val set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(msg)
    logging.info(msg)


def main(logging, dataset_aug, clc_num, parser):
    # Training settings
    args = parser.parse_args()
    utils.init_seed(args.seed)
    logging.info(args)
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'shuffle': True},)


    Config = SConfig()
    # load_data = load_shrec_data


    if dataset_aug == True:
        data_set = 'aug_train'
    else:
        data_set = 'train'

    train_data_generator = Sdata_generator(data_set, clc_num) #aug
    test_data_generator = Sdata_generator('val', clc_num)

    C = Config
    best_acc = 0
    best_epoch = 0
    # Train, Test = load_data()
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

    trainset = TensorDataset(X_0, X_1, X_2, Y)
    train_loader = DataLoader(trainset, **kwargs)

    testset = TensorDataset(X_0_t, X_1_t, X_2_t, Y_t)
    test_loader = DataLoader(
        testset, batch_size=args.test_batch_size)


    # Net = DSTANet(config=config)#
    Net = Dylan_MT_Net(8, args.mid_layer, clc_num, n_layers=args.net_layer)
    model = Net.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, amsgrad=False) # 0.001
    # optimizer = SGD(model.parameters(),lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    criterion = nn.CrossEntropyLoss()

    step = [30, 90]

    # scheduler = MultiStepLR(optimizer, step, gamma=args.gamma)
    # scheduler = CosineAnnealingLR(optimizer, lr_args.max_epoch + 1, eta_min=5e-6)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=args.gamma, patience=args.patience, cooldown=0, min_lr=args.min_lr, verbose=True) #5e-6

    # scheduler = utils.GradualWarmupScheduler(optimizer, total_epoch=args.warm_up_epoch,
    #                                          after_scheduler=lr_scheduler_pre)


    for epoch in range(1, args.epochs + 1):
        print('Epoch:', epoch)
        train_loss = train(args, model, device, train_loader,
                           optimizer, epoch, criterion, logging)
        test_val(model, device, train_loader, logging, criterion)
        acc = test(model, device, test_loader, logging, criterion)
        scheduler.step(train_loss)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
        print('Best acc===', best_acc,'at epoch:', best_epoch)



if __name__ == '__main__':
    sys.path.insert(0, './pytorch-summary/torchsummary/')
    savedir = Path('experiments') / Path(str(int(time.time())))
    print(savedir)
    makedir(savedir)
    logging.basicConfig(filename=savedir / 'train.log', level=logging.INFO)
    history = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    dataset_aug = False
    clc_num = 28
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=199, metavar='N',
                        help='number of epochs to train (default: 199)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--min_lr', type=float, default=5e-6, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--patience', type=int, default=5, metavar='M',
                        help='Learning rate step patience (default: 10)')
    parser.add_argument('--weight_decay', type=float, default=0.001, metavar='M',  # 0.001
                        help='weight_decay (default: 0.0005)')
    parser.add_argument('--net_layer', type=int, default=4)
    parser.add_argument('--mid_layer', type=int, default=4)
    parser.add_argument('--warm_up_epoch', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    main(logging, dataset_aug, clc_num, parser)
