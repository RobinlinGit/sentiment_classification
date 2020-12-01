import pandas as pd
from utils import text_preprocess, seg_char

df = pd.read_csv("./data/ai_competition_train_data.csv")
df["content"] = text_preprocess(df["content"], cut=seg_char)
df.to_csv("./data/char.train.csv", index=False)
# df = pd.read_csv("./data/sentiment_analysis_validationset.csv")
# df["content"] = text_preprocess(df["content"], cut=seg_char)
# df.to_csv("./data/char.valid.csv", index=False)

# df = pd.read_csv("./data/kg-test-no-lalel.csv")
# df["content"] = text_preprocess(df["content"], cut=seg_char)
# df.to_csv("./data/char.test.csv", index=False)