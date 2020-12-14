import json
import numpy as np
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
        self.pool_kernal = 4
        self.dim_after_pool = int(np.ceil((self.hid_dim * 2 - self.pool_kernal) / self.pool_kernal) + 1)
        self.aspect_dim = 128
        

class BilstmAspectAttPool(nn.Module):
    def __init__(self, configs=Configs()):
        super(BilstmAspectAttPool, self).__init__()
        self.emb = nn.Embedding(configs.voc_num, configs.emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=configs.emb_dim,
            hidden_size=configs.hid_dim,
            num_layers=configs.num_layers,
            dropout=configs.dropout,
            bidirectional=True)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool1d(4)
        self.fc0 = nn.Linear(configs.dim_after_pool, configs.aspect_dim)
        self.aspects = nn.Parameter(torch.randn(configs.aspect_dim, configs.aspect_num))
        self.fc1 = nn.Linear(configs.dim_after_pool + configs.aspect_dim, configs.fc1_dim)
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
        output = self.max_pool(output)
        output = self.tanh(output)
        # x [src len, batch size, hid dim2]
        output2 = self.fc0(output)
        # output [src len, batch size, aspect dim]
        alpha = torch.matmul(output2, self.aspects)
        alpha = F.softmax(alpha, dim=0)
        # alpha [src len, batch size, hid dim]
        alpha = alpha.unsqueeze(-2)
        # alpha [src len, batch size, 1, aspect num]
        output = output.unsqueeze(-1)
        # output [src len, batch size, hid dim, 1]
        output = output * alpha
        # output [src len, batch size, hid dim, aspect num]
        output = torch.sum(output, dim=0).permute(0, 2, 1)
        # output [batch size, aspect num, hid dim 2]
        batch_size = output.size()[0]
        aspects = self.aspects.unsqueeze(0).repeat(batch_size, 1, 1)
        # aspects [batch size, aspect dim, aspect num]
        aspects = aspects.permute(0, 2, 1)
        # aspects [batch size, aspect num, aspect dim]
        output = torch.cat([aspects, output], dim=2)
        output = F.relu(output, inplace=True)
        # classify
        output = self.fc1(output)
        output = F.relu(output, inplace=True)
        # output [batch size, aspect num, fc1 dim]
        output = self.fc2(output)
        return output

