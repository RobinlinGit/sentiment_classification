import pandas as pd
from utils import Voc


voc = Voc()
df = pd.read_csv("./data/char.train.csv")
voc.add(df["content"])
voc.dumps("./data/voc.json")