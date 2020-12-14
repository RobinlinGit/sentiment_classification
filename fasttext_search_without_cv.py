#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fasttext_search_without_cv.py
@Time    :   2020/11/25 15:09:24
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   fasttext 搜索参数，在验证集上验证，不带cv
'''

# %%
from numpy.lib.function_base import average
import pandas as pd
import os
from functools import reduce
from time import time
import json
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from models.fasttext_model import FasttextModel


columns = {
    "location": ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find'],
    "service": ['service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed',],
    "price": ['price_level', 'price_cost_effective', 'price_discount',],
    "environment": ['environment_decoration', 'environment_noise', 'environment_space', 'environment_clean',],
    "dish": ['dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',],
    "others": ['others_overall_experience', 'others_willing_to_consume_again']
}
all_columns = [columns[i] for i in columns]
all_columns = reduce(lambda a, b: a + b, all_columns)
all_columns = list(all_columns)
columns["all"] = all_columns
print(all_columns)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("column_type", choices=list(columns.keys()), type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    folder = "./temp"
    column = columns[args.column_type]
    filename = "./data/preprocessed.train.csv"
    valid_file = "./data/preprocessed.valid.csv"
    params_folder = "./params/fasttext"
    if not os.path.exists(params_folder):
        os.makedirs(params_folder)
    
    df = pd.read_csv(filename)
    valid_df = pd.read_csv(valid_file)
    X_val = valid_df["content"]
    X = df["content"]
    
    for c in column:
        
        print("*"*20 + f"{c}" + "*"*25)
        y_val = valid_df[c].values
        y = df[c].values

        # %%
        params = {
            "lr": [0.5, 0.7, 1],
            "ngrams": [2, 3, 4, 5],
            "min_count": [1, 2, 3, 4, 5, 8, 10],
            "ws": [4, 5, 6, 7],
            "dim": [50, 100, 150],
            "epoch": [20, 30, 40],
            "folder": ["./temp"]
        }
        record = pd.DataFrame(columns=list(params.keys()) + ["score"])
        params_importance = ["epoch", "ngrams", "min_count", "ws", "dim", "lr"]

        best_params = {
            "folder": "./temp"
        }
        for p in params_importance:
            temp_params = best_params.copy()
            search_range = params[p]
            score_rec = {}
            for v in search_range:
                
                temp_params[p] = v
                fastmodel = FasttextModel(**temp_params)
                fastmodel.fit(X, y)
                y_pred = fastmodel.predict(X_val)
                f1 = f1_score(y_val, y_pred, average="macro")
                rec = fastmodel.get_params()
                rec["score"] = f1
                score_rec[v] = f1
                record = record.append(rec, ignore_index=True)
                print(record)
            best_v = max(score_rec, key=score_rec.get)
            best_params[p] = best_v
            print(f"fasttext model on label {c}, best {p} is {best_v}")
        record.to_csv(os.path.join("./logs2/", f"{c}-{len(os.listdir('./logs2'))}.csv"))

if __name__ == "__main__":
    main()
