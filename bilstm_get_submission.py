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
voc = Voc()
voc.loads("./data/voc.json")
df = pd.read_csv("./data/char.test.csv")
columns = df.columns[-20:]
pred_df = df[columns].copy()
# %%
output_list = []
y_list = []

# %%
for i, content in tqdm(enumerate(df["content"])):
    seq = voc.sentence2idx(content.split(" "))
    seq_len = torch.LongTensor([len(seq)])
    seq = torch.LongTensor(seq)
    seq = seq.unsqueeze(-1)
    seq = seq.to(device)
    
    output = model(seq, seq_len)
    # print(output.size())
    output = output.to(torch.device("cpu"))
    output = output.argmax(-1)
    
    output_list.append(output)
# %%
output = torch.cat(output_list).numpy() - 2

columns = [
        'location_traffic_convenience',
        'location_distance_from_business_district', 'location_easy_to_find',
        'service_wait_time', 'service_waiters_attitude',
        'service_parking_convenience', 'service_serving_speed', 'price_level',
        'price_cost_effective', 'price_discount', 'environment_decoration',
        'environment_noise', 'environment_space', 'environment_clean',
        'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
        'others_overall_experience', 'others_willing_to_consume_again'
]
# %%
pred_df = pd.DataFrame(output, columns=columns)
pred_df.to_csv("./data/bilstm-pred.csv")

# %%
