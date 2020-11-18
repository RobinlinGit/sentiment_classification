#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fasttext_model.py
@Time    :   2020/11/18 15:15:19
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   将fasttext模型包装为scikit-learn的estimator
'''
import fasttext
import os
from sklearn.base import BaseEstimator, ClassifierMixin

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import df2txt


class FasttextModel(BaseEstimator, ClassifierMixin):

    def __init__(self, lr=1, ngrams=1, ws=5, dim=100, epoch=20, folder="./temp"):
        self.lr = lr
        self.ngrams = ngrams
        self.ws = ws
        self.dim = dim
        self.epoch = epoch
        self.folder = folder
    
    def fit(self, X, y):
        filename = os.path.join(
            self.folder,
            f"{self.lr}-{self.ngrams}-{self.ws}-{self.dim}-{self.epoch}.txt"
        )
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        df2txt(X, y, filename)
        # train fasttext model
        self.model_ = fasttext.train_supervised(
            input=filename,
            lr=self.lr,
            wordNgrams=self.ngrams,
            ws=self.ws,
            dim=self.dim,
            epoch=self.epoch
        )

    def predict(self, X, y=None):
        y = [0 for x in X]
        return y
    
    def get_params(self, deep=True):
        return {
            "lr": self.lr,
            "ngrams": self.ngrams,
            "ws": self.ws,
            "dim": self.dim,
            "epoch": self.epoch,
            "folder": self.folder
        }
        



