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

# first load stop words
with open("./configs/stopwords.txt", "r", encoding="utf-8") as f:
    stop_words = [x.replace('/n', '') for x in f.readlines()]


def text_preprocess(contents):
    def replace(x):
        x = x.replace('"', "").replace("\r\n", " ").replace("\n", " ")
        x = HanziConv.toSimplified(x)
        x = [a for a in jieba.cut(x) if a not in stop_words]
        x = " ".join(x)
        return x
    result = [replace(x) for x in tqdm(contents, desc="text preprocessing")]
    return result


def df2trainfile(df: pd.DataFrame, folder):
    columns = [x for x in df.columns[-20:]]
    # make sure the contents are preprocessed
    contents = df["content"]
    if not os.path.exists(folder):
        os.makedirs(folder)
    for n in columns:
        labels = df[n]
        filename = os.path.join(folder, f"{n}.txt")
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