#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2020/11/16 00:17:04
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   utils function
'''
import os
import pandas as pd
from tqdm import tqdm
import jieba
from hanziconv import HanziConv
from sklearn.metrics import f1_score
import re
import json
import torch
from torch.nn.utils.rnn import pad_sequence


PAD_token = 0
SOS_token = 1
UNK_token = 2
PAD = "<PAD>"
SOS = "<SOS>"
UNK = "<UNK>"


# first load stop words
with open("./configs/stopwords.txt", "r", encoding="utf-8") as f:
    stop_words = [x.replace('/n', '') for x in f.readlines()]


def seg_char(sent):
    """
    把句子按字分开，不破坏英文结构
    """
    # 首先分割 英文 以及英文和标点
    pattern_char_1 = re.compile(r'([\W])')
    parts = pattern_char_1.split(sent)
    parts = [p for p in parts if len(p.strip())>0]
    # 分割中文
    result = []
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    for p in parts:
        chars = pattern.split(p)
        chars = [w for w in chars if len(w.strip())>0]
        result += chars
    return result


def text_preprocess(contents, cut=jieba.cut):
    def replace(x):
        x = x.replace('"', "").replace("\r\n", " ").replace("\n", " ")
        x = HanziConv.toSimplified(x)
        x = [a for a in cut(x) if a not in stop_words]
        x = " ".join(x)
        return x
    result = [replace(x) for x in tqdm(contents, desc="text preprocessing")]
    return result


def df2trainfile(df: pd.DataFrame, folder, mode="train"):
    columns = [x for x in df.columns[-20:]]
    # make sure the contents are preprocessed
    contents = df["content"]
    if not os.path.exists(folder):
        os.makedirs(folder)
    for n in columns:
        labels = df[n]
        filename = os.path.join(folder, f"{n}.mode")
        df2txt(contents, labels, filename)


def df2txt(contents, labels, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for i, text in tqdm(enumerate(contents), desc=filename):
            f.write(f"__label__{labels[i]}\t{text}\n")


def get_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[1, 0, -1, -2], average='macro')


def result2submission(filename):
    df = pd.read_csv(filename)
    columns = df.columns[-20:]
    result = {"id": [], "label": []}
    for idx in range(len(df)):
        for i, c in enumerate(columns):
            s = df[c][idx]
            result["id"].append(f"{idx}-{i}")
            result["label"].append(s)
    df2 = pd.DataFrame(result)
    df2.to_csv("submission.csv", index=False)


class Voc(object):

    def __init__(self, min_count=5):
        self.word2idx = {PAD: PAD_token, SOS: SOS_token, UNK: UNK_token}
        self.idx2word = {}
        self.word2count = {}
        self.wordnum = 2
        self.min_count = min_count
    
    def add_sentence(self, sentence):
        """
        Args:
            sentence (list [str]).
        """
        for w in sentence:
            if w in self.word2idx:
                self.word2count[w] += 1
            else:
                self.word2idx[w] = self.wordnum
                self.wordnum += 1
                self.word2count[w] = 1
    
    def add(self, contents):
        for sentence in tqdm(contents):
            sentence = sentence.split(" ")
            self.add_sentence(sentence)
        self.trim(self.min_count)

    def trim(self, min_count):
        keep_words = [x for x, v in self.word2count.items() if v >= min_count]
        offset = 3
        self.word2idx = {x: i + offset for i, x in enumerate(keep_words)}
        self.word2idx[SOS] = SOS_token
        self.word2idx[PAD] = PAD_token
        self.word2idx[UNK] = UNK_token
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.wordnum = len(self.word2idx)
    
    def sentence2idx(self, sentence):
        sentence = map(lambda x: x if x in self.word2idx else UNK, sentence)
        index_list = [self.word2idx[x] for x in sentence]
        return index_list

    def dumps(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json.dumps(
                {
                    "word2idx": self.word2idx,
                    "idx2word": self.idx2word
                },
                ensure_ascii=False
            ))
    
    def loads(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
    
    def vocab_size(self):
        return len(self.word2idx)


# 加载数据集
class TextDataset(torch.utils.data.Dataset):
    
    def __init__(self, filename, voc_file):
        self.table = pd.read_csv(filename)
        self.label_columns = list(self.table.columns)[-20:]
        self.voc = Voc()
        self.voc.loads(voc_file)
    
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, index):
        content = [SOS] + self.table.iloc[index]["content"].split(" ")
        index_list = self.voc.sentence2idx(content)
        labels = self.table.iloc[index][self.label_columns].values.astype(int)
        return torch.LongTensor(index_list), torch.LongTensor(labels)

        
def pad_collate(batch):
    (xx, yy) = zip(*batch)

    x_lens = torch.Tensor([len(x) for x in xx])

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0).permute(1, 0)
    x_lens, per_idx = x_lens.sort(0, descending=True)
    yy = torch.cat([y.unsqueeze(0) for y in yy])
    xx_pad = xx_pad[:, per_idx]
    yy = yy[per_idx]
    yy += 2
    return xx_pad, yy, x_lens