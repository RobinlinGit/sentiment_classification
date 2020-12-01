import pandas as pd
from utils import get_f1_score

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

df = pd.read_csv("./data/char.valid.pred.csv")
df2 = pd.read_csv("./data/char.valid.csv")
print(df2.columns)
for c in columns:
    y_true = df2[c].values
    y_pred = df[c].values
    print(f"{c}: {get_f1_score(y_true, y_pred)}")