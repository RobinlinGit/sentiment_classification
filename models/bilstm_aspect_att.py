import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Configs(object):

    def __init__(self):
        self.hid_dim = 256
        self.voc_num = 7075
        self.aspect_num = 20
        self.emb_dim = 128
        self.fc1_dim = 128
        self.fc2_dim = 4    # class num
        self.dropout = 0.5
        self.num_layers = 1


class BilstmAspectAtt(nn.Module):

    def __init__(self, configs=Configs()):
        super(BilstmAspectAtt, self).__init__()
        self.emb = nn.Embedding(configs.voc_num, configs.emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=configs.emb_dim,
            hidden_size=configs.hid_dim,
            num_layers=configs.num_layers,
            dropout=configs.dropout,
            bidirectional=True)
        self.tanh = nn.Tanh()
        self.aspects = nn.Parameter(torch.randn(configs.hid_dim * 2, configs.aspect_num))
        self.fc1 = nn.Linear(2 * configs.hid_dim, configs.fc1_dim)
        self.fc2 = nn.Linear(configs.fc1_dim, configs.fc2_dim)
    
    def forward(self, x, x_lens):
        # x [src_len, batch_size]
        # x_lens [len0, len1, ...]
        x = self.emb(x)
        # print(f"x size: {x.size()}")
        # x [src len, batch size, emb dim]
        x_packed = pack_padded_sequence(x, x_lens, enforce_sorted=False)
        # print(x_packed.data.size())
        # print(x_packed)
        output, (_, _) = self.lstm(x_packed)
        # print(output.data.size())
        output, output_len = pad_packed_sequence(output)
        # print(f"lstm output: {output.size()}")
        # print(f"output_len: {output_len}")
        output = self.tanh(output)
        # x [src len, batch size, hid dim]
        alpha = torch.matmul(output, self.aspects)
        alpha = F.softmax(alpha, dim=0)
        # alpha [src len, batch size, hid dim]
        alpha = alpha.unsqueeze(-2)
        # alpha [src len, batch size, 1, aspect num]
        output = output.unsqueeze(-1)
        # output [src len, batch size, hid dim, 1]
        output = output * alpha
        # output [src len, batch size, hid dim, aspect num]
        output = torch.sum(output, dim=0).permute(0, 2, 1)
        # output [batch size, aspect num, hid dim]
        output = F.relu(output, inplace=True)
        # classify
        output = self.fc1(output)
        output = F.relu(output, inplace=True)
        # output [batch size, aspect num, fc1 dim]
        output = self.fc2(output)
        return output
        


