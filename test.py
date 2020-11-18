# %%
import pandas as pd

s = ["./data/char.train.csv", "./data/char.valid.csv"]
p = [pd.read_csv(x) for x in s]
df = pd.concat(p)
df.to_csv("./data/char.total.csv")