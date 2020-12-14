import pandas as pd
# import pkuseg
from utils import df2txt, text_preprocess, seg_char
df = pd.read_csv("./data/kg-test-no-lalel-update.csv")
df["content"] = text_preprocess(df["content"])
print(len(df))
df.to_csv("./data/jieba.test.csv", index=False)
