import math
import torch
import numpy as np
import torch.nn as nn

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


def hand_reshape(points_data, frame_l):
    Joint_point = np.array([[5, 9, 13, 17, 21],
                            [4, 8, 12, 16, 20],
                            [3, 7, 11, 15, 19],
                            [2, 6, 10, 14, 18],
                            [1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0]])
    points_data= points_data.reshape(frame_l, 3, 22)
    point = np.empty((frame_l, 3, 5, 6))
    for column in range(0, Joint_point.shape[0]):
        data_tmp = points_data[:, :, Joint_point[column]]
        point[:, :, :, column] = data_tmp.reshape(frame_l, 3, 5)
    return point


# ske = torch.rand([64, 12, 22])
# pe2 = positionalencoding2d(12, 6, 5)


def pe_2D(data, pe):
    tmp = torch.zeros(data.shape)
    Joint_point = np.array([[5, 9, 13, 17, 21],
                            [4, 8, 12, 16, 20],
                            [3, 7, 11, 15, 19],
                            [2, 6, 10, 14, 18],
                            [np.nan, np.nan, 1, np.nan, np.nan],
                            [np.nan, np.nan, 0, np.nan, np.nan]])

    for i in range(0, data.shape[1]):
        line = np.where(Joint_point == i)
        # print(i, line, pe[:, line[0], line[1]].shape)
        # print(data[:, :, i].shape)
        tmp[:, :, i] = data[:, :, i] + pe[:, line[0], line[1]].squeeze()
    return tmp


# print(type(pe_2D(ske, pe2)))

# print(Joint_point + Joint_point[0, 0])

conv = nn.Conv1d(231, 128, 1)

z = torch.rand([22, 231, 64])

print(conv(z).shape)
