# %%
import argparse
import fasttext
import os
import pandas as pd
from utils import result2submission, get_f1_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["valid", "test"], type=str, default="test")
    parser.add_argument("filename", type=str)
    parser.add_argument("model_folder", type=str)
    parser.add_argument("dst", type=str, help="result csv file folder")
    return parser.parse_args()


def main():
    args = parse_args()
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
    df = pd.read_csv(args.filename)
    folder = args.model_folder
    for c in columns:
        print(c)
        model = fasttext.load_model(os.path.join(folder, f"{c}.bin"))
        def predict(x):
            x = model.predict(x)[0][0]
            x = int(x[9:])
            return x
        df[c] = df["content"].apply(predict)
    df.to_csv(args.dst)
    if args.mode == "test":
        result2submission(args.dst)


if __name__ == "__main__":
    main()
