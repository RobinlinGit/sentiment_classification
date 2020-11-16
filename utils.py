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
        if mode == "train":
            filename = os.path.join(folder, f"{n}.txt")
        else:
            filename = os.path.join(folder, f"{n}.valid")
        with open(filename, "w", encoding="utf-8") as f:
            for i, text in tqdm(enumerate(contents), desc=f"{n}"):
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