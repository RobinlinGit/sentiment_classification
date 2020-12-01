#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   seq2seq_train_utils.py
@Time    :   2020/10/18 22:12:08
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   seq2seq 训练工具函数集合
'''
from tqdm import tqdm
import json
import random
import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, iterator, optimizer, criterion, clip, device):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in tqdm(enumerate(iterator), desc="trainning"):
        
        seqs, y, seq_lens = batch
        print(seqs.size())
        seqs = seqs.to(device)

        y = y.to(device)
        optimizer.zero_grad()
        output = model(seqs, seq_lens)
                
        #output = [batch size, 20, class num]
        #trg = [batch size, 4]
            
        output = output.contiguous().view(-1, 4)
        y = y.contiguous().view(-1)
        
        loss = criterion(output, y)
        output = output.to(torch.device("cpu"))
        y = y.to(torch.device("cpu"))
        print(loss)
        try:
            loss.backward()
        except RuntimeError as e:
            print(e)
            # print(loss)
            
            torch.save(output, "output.pt")
            torch.save(y, "y.pt")
            return
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in tqdm(enumerate(iterator), desc="test"):

            seqs, y, seq_lens = batch
            seqs = seqs.to(device)
            y = y.to(device)
            output = model(seqs, seq_lens)
                    
            #output = [batch size, 20, class num]
            #trg = [batch size, 4]

            output = output.contiguous().view(-1, 4)
            y = y.contiguous().view(-1)
                    
            loss = criterion(output, y)

            
            epoch_loss += loss.item()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
    return epoch_loss / len(iterator)
