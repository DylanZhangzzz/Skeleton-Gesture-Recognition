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
from MODEL.transformer import TransformerBlock
from torchsummary import summary

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
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)
        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        # pe = position/position.max()*2 -1
        # pe = pe.view(time_len, joint_num).unsqueeze(0).unsqueeze(0)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        print('pe_out', pe.dtype)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x




class Dylan_MT_Net(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, num_node=22, num_frame=64, hidden=384, n_layers=6, attn_heads=4, dropout=0.1):
        super(Dylan_MT_Net, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        num_channel = 3 # rgb
        # in_channels: word embedding size
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.fc = nn.Linear(self.out_channels, num_class)
        self.drop_out = nn.Dropout(dropout)

        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        # Position Encoding
        self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')
        self.pet = PositionalEncoding(out_channels, num_node, num_frame, 'temporal')
        # self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        # self.dropout = nn.Dropout(p=dropout)
        # 多个Encoder Layer叠加
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 2, dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(in_channels, eps=1e-6)

        self.temporal_blocks = nn.ModuleList(
            [TransformerBlock(num_node*out_channels, attn_heads, hidden * 4, dropout) for _ in range(3)])
        self.layer_norm = nn.LayerNorm(in_channels, eps=1e-6)


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        B, C, F, S = x.shape
        # x = x.permute(0, 3, 1, 2).contiguous()
        # print('input', x.size())
        x = self.input_map(x) # Batch, rgb, frames, skeleton = B, S, FC
        # print(x.size())
        # position encoding x = pe(x)
        x = self.pes(x)
        # data view for att
        x = x.permute(0, 3, 1, 2).contiguous().view(B, S, F*self.in_channels)# B, S, FC
        # print(x.size())
        for transformer in self.transformer_blocks:
            x = transformer.forward(x)

        x = x.reshape(B, S, F, self.in_channels).permute(0, 3, 2, 1) # B,C, F, S
        x = self.out_nets(x)
        # print('att_out1', x.size())
        # pe
        y=x
        y = self.pet(y)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, F, S*self.out_channels) # B, F, S*C

        for temporal_net in self.temporal_blocks:
            y = temporal_net.forward(y)
        y = y.view(B, F, S, self.out_channels).permute(0, 3, 1, 2)


        # print('out', y.size())

        z = Fnc.adaptive_avg_pool2d(y, (1, 1)).squeeze()
        z = self.drop_out(z)
        return self.fc(z)





if __name__ == '__main__':
    config = [[64, 64, 16], [64, 64, 16],
              [64, 128, 32], [128, 128, 32],
              [128, 256, 64], [256, 256, 64],
              [256, 256, 64], [256, 256, 64],
              ]
    net = Dylan_MT_Net(6, 128, 28)  # .cuda()
    # print(config[0][0])
    # print(net)
    ske = torch.rand([20, 64, 22, 3])  # .cuda() [batch, c, frame, skeleton] = B,N,T*C
    print(net(ske).shape)
    summary(net, [(64, 22, 3)])

