# %%
import pandas as pd
from sklearn.model_selection import GridSearchCV
from models.fasttext_model import FasttextModel

# %%
lr = 1
epoch = 20
ws = 5
ngrams = 1
dim = 100
folder = "./temp"

filename = "./data/char.total.csv"
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
y = df["price_level"].values

# %%
params = {
    "lr": [0.5]
}
fastmodel = FasttextModel()
model = GridSearchCV(
    fastmodel,
    param_grid=params,
    scoring="f1_macro",
    cv=5
)
model.fit(X, y)



# %%
