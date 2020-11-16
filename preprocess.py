import pandas as pd
import pkuseg
from utils import text_preprocess, seg_char

seg = pkuseg.pkuseg(model_name="web")
df = pd.read_csv("./data/sentiment_analysis_validationset.csv")
df["content"] = text_preprocess(df["content"], cut=seg_char)
df.to_csv("./data/char.valid.csv", index=False)
df = pd.read_csv("./data/sentiment_analysis_validationset.csv")
df["content"] = text_preprocess(df["content"], cut=seg.cut)
df.to_csv("./data/pkuseg.valid.csv", index=False)
df = pd.read_csv("./data/sentiment_analysis_validationset.csv")
df["content"] = text_preprocess(df["content"])
df.to_csv("./data/preprocessed.valid.csv", index=False)
# df = pd.read_csv("./data/kg-test-no-lalel.csv")
# df["content"] = text_preprocess(df["content"], cut=seg_char)
# df.to_csv("./data/char.test.csv", index=False)