#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fasttext_train.py
@Time    :   2020/11/16 11:55:33
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   train 20 fasttext model
'''
import os
import fasttext


lr = 1
minCount = 1
wordNgrams = 4
dim = 50
ws = 6


folder = "fasttext-data-char"
columns = [
    # 'location_traffic_convenience',
    # 'location_distance_from_business_district', 'location_easy_to_find',
    # 'service_wait_time', 'service_waiters_attitude',
    # 'service_parking_convenience', 'service_serving_speed', 'price_level',
    # 'price_cost_effective', 'price_discount', 'environment_decoration',
    # 'environment_noise', 'environment_space', 
     'environment_clean',
    'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
    'others_overall_experience', 'others_willing_to_consume_again'
]
model_folder = f"./model-zoo/fasttext-models-char-{lr}-{minCount}-{wordNgrams}-{dim}"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
for c in columns:
    print(c)
    c = "dish_taste"
    filename = os.path.join(folder, f"{c}.txt")
    valid = os.path.join(folder, f"{c}.valid")
    model = fasttext.train_supervised(
        input=filename,
        verbose=3,
        epoch=20,
        lr=lr,
        minCount=minCount,
        wordNgrams=wordNgrams,
        dim=dim
    )
    model.save_model(os.path.join(model_folder, f"{c}.bin"))
    break