# %%
import pandas as pd
from collections import Counter
from utils import text_preprocess

# %%
filename = "./submission.csv"
df = pd.read_csv(filename)
counter = Counter(df["label"])
print(counter)
# columns = df.columns[-20:]

# # %%
# result = {"id": [], "label": []}
# print(columns)

# # %%
# for idx in range(len(df)):
#     for i, c in enumerate(columns):
#         s = df[c][idx]
#         result["id"].append(f"{idx}-{i}")
#         result["label"].append(s)
# # %%
# df2 = pd.DataFrame(result)
# df2.to_csv("submission.csv", index=False)
# # %%
