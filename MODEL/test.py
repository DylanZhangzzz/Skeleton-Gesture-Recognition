import torch
import torch.nn as nn
from MODEL.transformer import TransformerBlock, MultiHeadedAttention, SublayerConnection

class Atten_Block(nn.Module):

    def __init__(self, attn_heads, hidden, dropout=0.1):
        super(Atten_Block, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x))

        return self.dropout(x)


if __name__ == '__main__':

    net = Atten_Block(16, 256)  # .cuda()
    # print(config[0][0])
    # print(net)
    # ske = torch.rand([20, 16, 22, 64])  # .cuda() [batch, c, frame, skeleton] = B,N,T*C [20, 3, 64, 22]
    # jcd = torch.rand([20, 16, 22, 32])
    # # print(net(ske, jcd).shape)
    # y = ske + jcd

