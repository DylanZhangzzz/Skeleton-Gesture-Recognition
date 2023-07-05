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


class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

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
            # spatial embedding
            # pos_list = []
            # for t in range(self.time_len):
            #     for j_id in range(self.joint_num):
            #         pos_list.append(j_id)
            tmp = torch.zeros(self.time_len, channel, self.joint_num)
            pe2 = utils.positionalencoding2d(channel, 6, 5)
            pe = utils.pe_2D(tmp, pe2).permute(1, 0, 2).unsqueeze(0).float()
            # print('pe_out', pe.dtype)
            self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        # print('pe', x.shape)
        x = x + self.pe[:, :, :x.size(2)]
        return x

class c1D(nn.Module):
    # input (B,C,D) //batch,channels,dims
    # output = (B,C,filters)
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(c1D, self).__init__()
        self.cut_last_element = (kernel % 2 == 0)
        self.padding = math.ceil((kernel - 1)/2)
        self.conv1 = nn.Conv1d(input_dims, filters,
                               kernel, bias=False, padding=self.padding)
        self.bn = nn.BatchNorm1d(num_features=input_channels)

    def forward(self, x):
        # x (B,D,C)
        x = x.permute(0, 2, 1)
        # output (B,filters,C)
        if(self.cut_last_element):
            output = self.conv1(x)[:, :, :-1]
        else:
            output = self.conv1(x)
        # output = (B,C,filters)
        output = output.permute(0, 2, 1)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2, True)
        return output


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
        self.layers = 1
        self.jcd_f = 231
        self.hidden = num_node*in_channels

        # Position Encoding
        self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')
        self.pet = PositionalEncoding(in_channels*self.layers, num_node, num_frame, 'temporal')

        self.s_att = Atten_Block(attn_heads, num_frame * self.in_channels, dropout)

        self.t_att = Atten_Block(attn_heads, self.hidden, dropout)

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        B, C, F, S = x.shape
        x = self.pes(x)
        # data view for att
        x = x.permute(0, 3, 1, 2).contiguous().view(B, S, F*self.in_channels)# B, S, FC
        # print('att_in', x.size())
        x = self.s_att(x)
        x = x.reshape(B, S, F, self.in_channels).permute(0, 3, 2, 1) # B,C, F, S

        y = x
        y = self.pet(y)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, F, S*self.in_channels*self.layers) # B, F, S*C
        # print('Y att_in', y.size())

        # for temporal_net in self.temporal_blocks:
        #     y = temporal_net.forward(y)
        y = self.t_att(y)
        y = y.view(B, F, S, self.in_channels*self.layers).permute(0, 3, 1, 2)
        # print('Y out', y.size())

        # z = Fnc.adaptive_avg_pool2d(y, (1, 1)).squeeze() #GAP
        # z = self.drop_out(z)
        return y #self.fc(z)


class MS_CAM(nn.Module):

    def __init__(self, channel, ratio = 4):
        super(MS_CAM, self).__init__()
        mid_channel = channel // ratio
        self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=channel, out_channels=mid_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=mid_channel, out_channels=channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channel),
            )

        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=mid_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channel, out_channels=channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        g_x = self.global_att(x)
        l_x = self.local_att(x)
        w = self.sigmoid(l_x * g_x.expand_as(l_x))
        return w * x

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


class Dylan_MT_Net(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, num_node=22, num_frame=64, n_layers=6, attn_heads=16, dropout=0.05):
        super(Dylan_MT_Net, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.layers = 1
        self.num_node = num_node
        self.num_frame = num_frame
        self.hidden = num_node * in_channels
        num_channel = 3  # rgb
        # in_channels: word embedding size
        self.input_map_1 = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.output_map = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )
        self.fc = nn.Linear(128, num_class)  # self.out_channels
        self.drop_out = nn.Dropout(dropout)

        self.att_blocks = nn.ModuleList(
            [MT_Net(in_channels, out_channels, num_node, num_frame, attn_heads, dropout) for _ in range(n_layers)])

        self.fusion_blocks = nn.ModuleList(
            [MS_CAM(in_channels) for _ in range(3)])
        # self.att_fusion = MS_CAM(self.num_frame)

        # self.st_block = MT_Net(in_channels, out_channels, num_class, num_node=22, num_frame=64, attn_heads=8, dropout=0.1)

        self.linear1 = nn.Sequential(
            d1D(self.in_channels*self.num_node*num_frame, 256),
            nn.Dropout(0.5)
        )
        self.linear2 = nn.Sequential(
            d1D(256, 128),
            nn.Dropout(0.5)
        )

    def forward(self, x, x2, x3):
        # in_channels: word embedding size
        x = x.permute(0, 3, 1, 2)
        B, C, F, S = x.shape
        # x = x.permute(0, 3, 1, 2).contiguous()
        # print('input', x.size())
        x = self.input_map_1(x)  # Batch, rgb, frames, skeleton = B, S, FC
        # position encoding x = pe(x)

        for att in self.att_blocks:
            x = att.forward(x)

        # print('in', x.shape)
        # x = x.permute(0, 3, 1, 2)
        # for fus in self.fusion_blocks:
        #     x = fus.forward(x)

        # x = self.att_fusion(x)
        # x = Fnc.adaptive_avg_pool2d(x, (1, 1)).squeeze()  # GAP
        # x = self.drop_out(x)
        # x = torch.max(x, dim=1).values.view(B, self.num_node*self.num_frame)
        # print(x.shape)
        # x = torch.max(x, dim=3).values.view(B, self.num_node * self.in_channels)
        x = torch.flatten(x, start_dim=1)
        # print('out', x.shape)
        x = self.linear1(x)
        x = self.linear2(x)

        return self.fc(x)







if __name__ == '__main__':
    config = [[64, 64, 16], [64, 64, 16],
              [64, 128, 32], [128, 128, 32],
              [128, 256, 64], [256, 256, 64],
              [256, 256, 64], [256, 256, 64],
              ]
    net = Dylan_MT_Net(16, 64, 28)  # .cuda()
    # print(config[0][0])
    # print(net)
    ske = torch.rand([20, 64, 22, 3])  # .cuda() [batch, c, frame, skeleton] = B,N,T*C [20, 3, 64, 22]
    jcd = torch.rand([20, 64, 231])
    print(net(ske, jcd, ske).shape)
    summary(net, [(64, 22, 3), (64, 231),(64, 22, 3)])

