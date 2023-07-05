from torchsummary import summary  # noqa
from pathlib import Path
import numpy as np
import utils
from utils import makedir
import argparse
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR, StepLR
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

#from MODEL.dstanet import DSTANet
from MODEL.dylan_net_v7 import Dylan_MT_Net
# from Dataloader.Shrec_dataset import load_shrec_data, Sdata_generator, SConfig
from Dataloader.skeleton_loader import SConfig, Sdata_generator
import torch

import torch.nn as nn


from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_net(parameterization):
    model = Dylan_MT_Net(3, 4,
                       28, num_node=22, num_frame=parameterization.get("num_frame", 120),
                       n_layers=2, attn_heads=6,
                       dropout=parameterization.get("att_drop", 0.0005),
                         l_dropout=parameterization.get("l_drop", 0.3))
    return model  # return untrained model


def net_train(net, train_loader, parameters, dtype, device):
    net.to(dtype=dtype, device=device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = SGD(net.parameters(),  # or any optimizer you prefer
    #                       lr=parameters.get("lr", 0.001),  # 0.001 is used if no lr is specified
    #                       momentum=parameters.get("momentum", 0.9)
    #                       )
    optimizer = Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=parameters.get("weight_decay", 0.005), amsgrad=False)

    # scheduler = StepLR(
    #     optimizer,
    #     step_size=int(parameters.get("step_size", 30)),
    #     gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
    # )
    scheduler = ReduceLROnPlateau(
        optimizer, factor=parameters.get("gamma", 0.1), patience=5, cooldown=0, min_lr=0, verbose=True)

    num_epochs = parameters.get("num_epochs", 3)  # Play around with epoch number
    # Train Network
    train_loss=0
    for _ in range(num_epochs):
        for inputs, labels in train_loader:
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.detach().item()
            loss.backward()
            optimizer.step()
            scheduler.step(train_loss)
    return net


def train_evaluate(parameterization):
    train_data_generator = Sdata_generator('train', 28)  # aug
    test_data_generator = Sdata_generator('val', 28)

    Config = SConfig(parameterization.get("num_frame", 120))
    best_acc = 0
    best_epoch = 0
    # Train, Test = load_data()
    X_0, X_1, X_2, Y = train_data_generator(Config, False)
    X_0 = torch.from_numpy(X_0).type('torch.FloatTensor')
    Y = torch.from_numpy(Y).type('torch.LongTensor')

    X_0_t, X_1_t, X_2_t, Y_t = test_data_generator(Config)
    X_0_t = torch.from_numpy(X_0_t).type('torch.FloatTensor')
    Y_t = torch.from_numpy(Y_t).type('torch.LongTensor')

    trainset = TensorDataset(X_0, Y)

    testset = TensorDataset(X_0_t, Y_t)
    test_loader = DataLoader(
        testset, batch_size=1000)
    # constructing a new training data loader allows us to tune the batch size
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=parameterization.get("batchsize", 32),
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    # Get neural net
    untrained_net = init_net(parameterization)

    # train
    trained_net = net_train(net=untrained_net, train_loader=train_loader,
                            parameters=parameterization, dtype=dtype, device=device)

    # return the accuracy of the model as it was trained in this run
    return evaluate(
        net=trained_net,
        data_loader=test_loader,
        dtype=dtype,
        device=device,
    )


# torch.cuda.set_device(0) #this is sometimes necessary for me
dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_parameters, values, experiment, model = optimize(
    parameters=[
        # {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
        {"name": "batchsize", "type": "range", "bounds": [4, 128]},
        # {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
        # {"name": "gamma", "type": "range", "bounds": [0.00005, 0.9]},
        # {"name": "weight_decay", "type": "range", "bounds": [0.00005, 0.9]},
        # {"name": "att_drop", "type": "range", "bounds": [0.00005, 0.5]},
        # {"name": "l_drop", "type": "range", "bounds": [0.00005, 0.5]},
        # {"name": "max_epoch", "type": "range", "bounds": [1, 30]},
        # {"name": "stepsize", "type": "range", "bounds": [20, 40]},
    ],
    total_trials=199,
    evaluation_function=train_evaluate,
    objective_name='accuracy',
)

print(best_parameters)
means, covariances = values
print(means)
print(covariances)

best_objectives = np.array([[trial.objective_mean * 100 for trial in experiment.trials.values()]])

best_objective_plot = optimization_trace_single_method(
    y=np.maximum.accumulate(best_objectives, axis=1),
    title="Model performance vs. # of iterations",
    ylabel="Classification Accuracy, %",
)
render(best_objective_plot)

render(plot_contour(model=model, param_x='batchsize', param_y='weight_decay', metric_name='accuracy'))
data = experiment.fetch_data()
df = data.df
best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
best_arm = experiment.arms_by_name[best_arm_name]
print(best_arm)