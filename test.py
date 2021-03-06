#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2020/12/01 19:48:05
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   bilstm模型测试结果
'''
# %%
from numpy.lib.function_base import average
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from utils import Voc, TextDataset, pad_collate
from models.bilstm_att_pool import BilstmAspectAttPool


class Configs0(object):

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


class Configs1(object):

    def __init__(self):
        self.hid_dim = 256
        self.voc_num = 7075
        self.aspect_num = 20
        self.emb_dim = 128
        self.fc1_dim = 64
        self.fc2_dim = 4    # class num
        self.dropout = 0.5
        self.num_layers = 2
        self.pool_kernal = 4
        self.dim_after_pool = int(np.ceil((self.hid_dim * 2 - self.pool_kernal) / self.pool_kernal) + 1)
        self.aspect_dim = 64


filename = "./data/char.valid.csv"
# %%
configs = Configs1()
model = BilstmAspectAttPool(configs)
model.load_state_dict(torch.load("./model-zoo/bilstm_aspect_att_pool2.pt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
testset = TextDataset(filename, "./data/voc.json")
test_loader = DataLoader(testset, batch_size=4, shuffle=False, collate_fn=pad_collate)

# %%
output_list = []
y_list = []

# %%
for batch in tqdm(test_loader):
    seq, y, seq_len = batch
    seq = seq.to(device)
    
    output = model(seq, seq_len)
    # print(output.size())
    output = output.to(torch.device("cpu"))
    output = output.argmax(-1)
    
    output_list.append(output)
    y_list.append(y)
# %%
output = torch.cat(output_list).numpy()
y = torch.cat(y_list).numpy()
df = pd.read_csv(filename)
columns = df.columns[-20:]
f1_list = []
for i in range(20):
    score = f1_score(y_true=y[:, i], y_pred=output[:, i], average="macro")
    f1_list.append(score)
    print(f"{columns[i]}: {score}")
print(f"total: {np.mean(f1_list)}")
