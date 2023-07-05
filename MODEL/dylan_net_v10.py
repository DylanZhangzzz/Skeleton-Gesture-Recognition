import sys
import os
parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as Fnc
from MODEL.transformer import TransformerBlock, MultiHeadedAttention, SublayerConnection
from torchsummary import summary
import utils
import torch.nn.functional as F

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len
        self.channel = channel

        self.domain = domain

        channel = channel + 1
        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
            position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
            pe = torch.zeros(self.time_len * self.joint_num, channel)

            div_term = torch.exp(torch.arange(0, channel, 2).float() *
                                 -(math.log(10000.0) / channel))  # channel//2
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
            # print('pe_out', pe.shape)
            self.register_buffer('pe', pe)

        elif domain == "spatial":
            tmp = torch.zeros(self.time_len, channel, self.joint_num)
            pe2 = utils.positionalencoding2d(channel, 6, 5)
            pe = utils.pe_2D(tmp, pe2).permute(1, 0, 2).unsqueeze(0).float()
            # print('pe_out', pe.shape)
            self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        # print('pe', x.shape)
        x = x + self.pe[:, :self.channel, :x.size(2)]
        return x



class Atten_Block(nn.Module):

    def __init__(self, attn_heads, hidden, dropout=0.1):
        super(Atten_Block, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x))

        return self.dropout(x)


class MT_Net(nn.Module):
    def __init__(self, in_channels, out_channels, num_node, num_frame, attn_heads, dropout):
        super(MT_Net, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.hidden = num_node*in_channels

        self.s_att = Atten_Block(attn_heads, num_frame * self.in_channels, dropout) # 16
        self.t_att = Atten_Block(attn_heads, num_node*self.in_channels, dropout) # 11

        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=dropout)

        # print('reset')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, data_x, data_y):
        B, C, F, S = data_x.shape
        x = data_x
        # data view for att
        x = x.permute(0, 3, 1, 2).contiguous().view(B, S, F*self.in_channels)# B, S, FC
        # print('att_in', x.size())
        x = self.s_att(x)
        x = x.reshape(B, S, F, self.in_channels).permute(0, 3, 2, 1) # B,C, F, S

        y = data_y
        y = y.permute(0, 2, 1, 3).contiguous().view(B, F, S*self.in_channels) # B, F, S*C
        # print('Y att_in', y.size())
        y = self.t_att(y)
        y = y.view(B, F, S, self.in_channels).permute(0, 3, 1, 2)
        # print('Y out', y.size())

        return self.dropout(x), self.dropout(y) #x, y



class d1D(nn.Module):
    def __init__(self, input_dims, filters):
        super(d1D, self).__init__()
        self.linear = nn.Linear(input_dims, filters)
        self.bn = nn.BatchNorm1d(num_features=filters)

    def forward(self, x):
        output = self.linear(x)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2)
        return output



class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels, inter_channels):
        super(iAFF, self).__init__()
        # inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


class Dylan_MT_Net(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, num_node=22, num_frame=64, n_layers=2, attn_heads=6, dropout=0.05, l_dropout=0.2):
        super(Dylan_MT_Net, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_node = num_node
        self.num_frame = num_frame
        self.hidden = num_node * in_channels
        self.pool_size = 2
        # self.fusion = iAFF(in_channels, out_channels)
        mid_layer = 1
        # in_channels: word embedding size
        self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')
        self.pet = PositionalEncoding(in_channels, num_node, num_frame, 'temporal')

        self.att_blocks = nn.ModuleList(
            [MT_Net(in_channels, out_channels, num_node, num_frame, attn_heads, dropout) for _ in range(n_layers)])

        self.pool_layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=(self.pool_size, 1)),
            nn.Dropout(l_dropout)
        )

        self.linear1 = nn.Sequential(
            d1D(self.in_channels*self.num_node*num_frame, 512*mid_layer), #self.in_channels*self.num_node*num_frame//2
            nn.Dropout(l_dropout)
        )
        self.linear2 = nn.Sequential(
            d1D(512*mid_layer, 256*mid_layer),
            nn.Dropout(l_dropout)
        )
        self.linear3 = nn.Sequential(
            d1D(256*mid_layer, 256*mid_layer),
            nn.Dropout(l_dropout)
        )

        self.linear4 = nn.Sequential(
            d1D(self.in_channels * self.num_node * num_frame, 512 * mid_layer),
            # self.in_channels*self.num_node*num_frame//2
            nn.Dropout(l_dropout)
        )
        self.linear5 = nn.Sequential(
            d1D(512 * mid_layer, 256 * mid_layer),
            nn.Dropout(l_dropout)
        )
        self.linear6 = nn.Sequential(
            d1D(256 * mid_layer, 256 * mid_layer),
            nn.Dropout(l_dropout)
        )
        self.fc = nn.Linear(256*mid_layer, num_class)  # self.out_channels # 128
        # self.gpa = nn.AdaptiveAvgPool2d((1,1))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         conv_init(m)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         bn_init(m, 1)
        #     elif isinstance(m, nn.Linear):
        #         fc_init(m)

    def forward(self, x, x2=None, x3=None):
        # in_channels: word embedding size
        x = x.permute(0, 3, 1, 2)
        # y = x
        # x = x.permute(0, 3, 1, 2).contiguous()
        # print('input', x.size())
        x = self.pes(x)
        # x = self.pet(x)
        y = self.pet(x)
        # print('input', x.size())
        for att in self.att_blocks:
            x, y = att.forward(x, y)

        # print('out', x.shape, y.shape)
        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        # print('out', x.shape, y.shape)
        # x = x.permute(0, 3, 1, 2)
        # y = y.permute(0, 3, 1, 2)
        # z = self.fusion(x, y)
        # print(x.shape)
        # z = torch.concat((x, y), dim=1)
        # print(z.shape)
        # z = torch.mean(z, dim= 1)
        # print(z.shape)
        # print('out', z.shape)
        # z = z.permute(0, 3, 1, 2)
        # print('out', z.shape)
        # z = self.gpa(z).squeeze()
        # print('in', z.shape)
        # z = self.pool_layer(z)
        # print('out', z.shape)
        # # print('flatten_out', z.shape)
        z = self.linear1(x)
        z = self.linear2(z)
        z = self.linear3(z)
        # # print('flatten_out', z.shape)
        y = self.linear1(y)
        y = self.linear2(y)
        y = self.linear3(y)
        # print('flatten_out', z.shape, y.shape)
        out = torch.add(z, y)
        out = torch.divide(out, 2)
        # print('flatten_out', out.shape)
        return self.fc(out)





if __name__ == '__main__':
    config = [[64, 64, 16], [64, 64, 16],
              [64, 128, 32], [128, 128, 32],
              [128, 256, 64], [256, 256, 64],
              [256, 256, 64], [256, 256, 64],
              ]
    net = Dylan_MT_Net(3, 6, 28)  # .cuda()
    # print(config[0][0])
    # print(net)
    ske = torch.rand([20, 64, 22, 3])  # .cuda() [batch, c, frame, skeleton] = B,N,T*C [20, 3, 64, 22]
    jcd = torch.rand([20, 64, 231])
    print(net(ske, jcd, ske).shape)
    summary(net, [(64, 22, 3), (64, 231),(64, 22, 3)])

