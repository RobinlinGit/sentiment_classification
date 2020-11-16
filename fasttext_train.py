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
from tqdm import tqdm


folder = "fasttext-data-char"
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
model_folder = "./models/fasttext-models-char"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
for c in columns:
    print(c)
    filename = os.path.join(folder, f"{c}.txt")
    model = fasttext.train_supervised(
        filename,
        epoch=20,
        wordNgrams=3,
        minCount=5,
        lr=1.0
    )
    model.save_model(os.path.join(model_folder, f"{c}.bin"))

