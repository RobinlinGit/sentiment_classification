# %%
import pandas as pd
import os
from time import time
import json
import argparse
from sklearn.model_selection import GridSearchCV
from models.fasttext_model import FasttextModel


columns = [
        'location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
        'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed',
        'price_level', 'price_cost_effective', 'price_discount',
        'environment_decoration', 'environment_noise', 'environment_space', 'environment_clean',
        'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
        'others_overall_experience', 'others_willing_to_consume_again'
    ]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("column", choices=columns, type=str)
    return parser.parse_args()


def main():
    # args = parse_args()
    folder = "./temp"
    # column = args.column
    column = [
       'others_overall_experience', 'others_willing_to_consume_again'
    ]
    filename = "./data/char.total.csv"
    params_folder = "./params/fasttext"
    if not os.path.exists(params_folder):
        os.makedirs(params_folder)
    
    df = pd.read_csv(filename)
    X = df["content"]
    for c in column:
        print("*"*20 + f"{c}" + "*"*25)
        y = df[c].values

        # %%
        params = {
            "lr": [0.5],
            "ngrams": [3, 4, 5],
            "min_count": [1, 2, 3, 4, 5, 8, 10],
            "ws": [4, 5, 6, 7],
            "dim": [50, 100, 150],
            "epoch": [20],
            "folder": ["./temp"]
        }
        params_importance = ["ngrams", "min_count", "ws", "dim"]

        best_params = {
            "lr": 0.5,
            "epoch": 20,
            "folder": "./temp"
        }
        for p in params_importance:
            temp_params = {k: [best_params[k]] for k in best_params}
            temp_params[p] = params[p]
            fastmodel = FasttextModel()
            model = GridSearchCV(
                fastmodel,
                param_grid=temp_params,
                scoring="f1_macro",
                cv=3,
                refit=False
            )
            start = time()
            model.fit(X, y)
            best_params = model.best_params_d
            
            # %%
            print(f"grid search on {p} cost time {(time() - start) / 60} min")
            print(model.best_params_)
            result_filename = os.path.join("./logs/", f"{c}-{len(os.listdir('./logs'))}.csv")
            r = pd.DataFrame(model.cv_results_)
            print(r)
            r.to_csv(result_filename)
        # %%
            with open(os.path.join(params_folder, f"{c}.json"), "w") as f:
                f.write(json.dumps(model.best_params_))


if __name__ == "__main__":
    main()