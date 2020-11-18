# %%
import pandas as pd
import os
from time import time
import json
from sklearn.model_selection import GridSearchCV
from models.fasttext_model import FasttextModel


# %%
folder = "./temp"
column = "price_level"
filename = "./data/char.total.csv"
params_folder = "./params/fasttext"
if not os.path.exists(params_folder):
    os.makedirs(params_folder)
columns = [
    'location_traffic_convenience',
    'location_distance_from_business_district', 'location_easy_to_find',
    'service_wait_time', 'service_waiters_attitude',
    'service_parking_convenience', 'service_serving_speed', 'price_level',
    'price_cost_effective', 'price_discount', 'environment_decoration',
    'environment_noise', 'environment_space',
     'environment_clean',
    'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
    'others_overall_experience', 'others_willing_to_consume_again'
]
df = pd.read_csv(filename)
X = df["content"]
y = df[column].values

# %%
params = {
    "lr": [0.5],
    "ngrams": [4],
    "min_count": [2],
    "ws": [4, 5, 6, 7],
    "dim": [50, 100, 150],
    "epoch": [20],
    "folder": ["./temp"]
}
params_importance = ["ws", "dim"]

best_params = {
    "lr": 0.5,
    "epoch": 20,
    "ngrams": 4,
    "min_count": 2,
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
    best_params = model.best_params_
    
    # %%
    print(f"grid search on {p} cost time {(time() - start) / 60} min")
    print(model.best_params_)
    result_filename = os.path.join("./logs/", f"{column}-{len(os.listdir('./logs'))}.csv")
    r = pd.DataFrame(model.cv_results_)
    print(r)
    r.to_csv(result_filename)
# %%
    with open(os.path.join(params_folder, f"{column}.json"), "w") as f:
        f.write(json.dumps(model.best_params_))