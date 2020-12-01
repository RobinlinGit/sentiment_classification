import pandas as pd
from utils import df2trainfile


df = pd.read_csv("./data/char.train.csv")
df2trainfile(df, "./fasttext-data-char")