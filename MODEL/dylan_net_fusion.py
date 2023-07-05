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


class spatialDropout1D(nn.Module):
    def __init__(self, p):
        super(spatialDropout1D, self).__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
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


class Dylan_MT_Net(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, num_node=22, num_frame=64, n_layers=6, attn_heads=8, dropout=0.1):
        super(Dylan_MT_Net, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.layers = 1
        self.jcd_f = 231
        self.jcd_hidden = num_node*in_channels
        num_channel = 3 # rgb
        # in_channels: word embedding size
        self.input_map_1 = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.input_map_jcd = nn.Sequential(
            nn.Conv1d(231, self.jcd_hidden, 1),
            nn.BatchNorm2d(self.jcd_hidden),
            nn.LeakyReLU(0.1),
        )
        # self.input_map_3 = nn.Sequential(
        #     nn.Conv2d(num_channel, in_channels, 1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.LeakyReLU(0.1),
        # )

        self.fc = nn.Linear(64, num_class) #self.out_channels
        self.drop_out = nn.Dropout(dropout)

        # Position Encoding
        self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')
        self.pet = PositionalEncoding(in_channels*self.layers, num_node, num_frame, 'temporal')
        # self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        # self.dropout = nn.Dropout(p=dropout)
        # 多个Encoder Layer叠加

        self.att_blocks = nn.ModuleList(
            [Atten_Block(attn_heads, num_frame*self.in_channels) for _ in range(n_layers)])

        self.temporal_blocks = nn.ModuleList(
            [Atten_Block(attn_heads, self.jcd_hidden) for _ in range(n_layers)])


        self.relu = nn.LeakyReLU(0.1)

        self.out_nett = nn.Sequential(
            nn.Conv2d(in_channels * self.layers, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.ff_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels),
        )

        self.jcd_conv1 = nn.Sequential(
            c1D(num_frame, self.jcd_f, 2 *  self.jcd_hidden, 1),
            spatialDropout1D(0.1)
        )
        self.jcd_conv2 = nn.Sequential(
            c1D(num_frame, 2 *  self.jcd_hidden,  self.jcd_hidden, 3),
            spatialDropout1D(0.1)
        )
        self.jcd_conv3 = c1D(num_frame,  self.jcd_hidden,  self.jcd_hidden, 1)
        self.jcd_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1)
        )

    def forward(self, x, x2, x3):
        x = x.permute(0, 3, 1, 2)
        # x = x2.permute(0, 3, 1, 2)
        # x3 = x3.permute(0, 3, 1, 2)
        B, C, F, S = x.shape
        # x = x.permute(0, 3, 1, 2).contiguous()
        # print('input', x.size())
        x = self.input_map_1(x) # Batch, rgb, frames, skeleton = B, S, FC
        # position encoding x = pe(x)
        x = self.pes(x)
        # data view for att
        x = x.permute(0, 3, 1, 2).contiguous().view(B, S, F*self.in_channels)# B, S, FC
        # print('att_in', x.size())
        for transformer in self.att_blocks:
            x = transformer.forward(x)
        x = x.reshape(B, S, F, self.in_channels).permute(0, 3, 2, 1) # B,C, F, S
        # print('att_out', x.size())
        # x = self.relu(self.out_nets_1(x))
        # print('att_out1', x.size())
        #x2
        # pe
        # x2 = self.input_map_2(x2)
        # x2 = self.pes(x2)
        # x2 = x2.permute(0, 3, 1, 2).contiguous().view(B, S, F * self.in_channels)  # B, S, FC
        #
        # for transformer in self.transformer_blocks_1:
        #     x2 = transformer.forward(x2)
        # x2 = x2.reshape(B, S, F, self.in_channels).permute(0, 3, 2, 1)  # B,C, F, S
        # x2 = self.relu(self.out_nets_2(x2))
        # # print('att_out2', x2.size())
        #
        # x3 = self.input_map_1(x3)  # Batch, rgb, frames, skeleton = B, S, FC
        # # position encoding x = pe(x)
        # x3 = self.pes(x3)
        # # data view for att
        # x3 = x3.permute(0, 3, 1, 2).contiguous().view(B, S, F * self.in_channels)  # B, S, FC
        # for transformer in self.transformer_blocks:
        #     x3 = transformer.forward(x3)
        # x3 = x3.reshape(B, S, F, self.in_channels).permute(0, 3, 2, 1)  # B,C, F, S
        # x3 = self.relu(self.out_nets_1(x3))
        # print('att_out3', x3.size())
        # j = self.jcd_conv1(x2)
        # j = self.jcd_conv2(j)
        # j = self.jcd_conv3(j)
        # print('input_j', j.size())

        # y = torch.cat((x, j), dim=1)
        y = x
        # print('Y att_in', y.size())
        y = self.pet(y)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, F, S*self.in_channels*self.layers) # B, F, S*C
        # print('input_y', y.size())
        # y = torch.cat((y, j), dim=2)

        for temporal_net in self.temporal_blocks:
            y = temporal_net.forward(y)

        y = y.view(B, F, S, self.in_channels*self.layers) #.permute(0, 3, 1, 2)


        z = Fnc.adaptive_avg_pool2d(y, (1, 1)).squeeze() #GAP
        z = self.drop_out(z)
        return self.fc(z)





if __name__ == '__main__':
    config = [[64, 64, 16], [64, 64, 16],
              [64, 128, 32], [128, 128, 32],
              [128, 256, 64], [256, 256, 64],
              [256, 256, 64], [256, 256, 64],
              ]
    net = Dylan_MT_Net(32, 128, 28)  # .cuda()
    # print(config[0][0])
    # print(net)
    ske = torch.rand([20, 64, 22, 3])  # .cuda() [batch, c, frame, skeleton] = B,N,T*C [20, 3, 64, 22]
    jcd = torch.rand([20, 64, 231])
    print(net(ske, jcd, ske).shape)
    summary(net, [(64, 22, 3), (64, 231),(64, 22, 3)])

