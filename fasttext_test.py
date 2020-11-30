# %%
import fasttext
import pandas as pd
from utils import result2submission


# %%
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
# df = pd.read_csv("./data/processed.test.csv")

# for c in columns:
#     print(c)
#     model = fasttext.load_model(f"./models/fasttext-models/{c}.bin")
#     def predict(x):
#         x = model.predict(x)[0][0]
#         x = int(x[9:])
#         return x
#     df[c] = df["content"].apply(predict)
# df.to_csv("./data/result.csv")
result2submission("./data/result.csv")
