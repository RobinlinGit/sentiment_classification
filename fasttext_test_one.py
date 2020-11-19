#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fasttext_test_one.py
@Time    :   2020/11/19 15:08:54
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   测试单个模型的效果
'''
import fasttext


column = "dish_taste"
model = f"./model-zoo/fasttext-models-char-1-1-4-100/{column}.bin"
valid_file = f"./fasttext-data-char/{column}.valid"
model = fasttext.load_model(model)
result = model.test(valid_file)
f1 = 2 * result[1] * result[2] / (result[1] + result[2])
print(f1)

