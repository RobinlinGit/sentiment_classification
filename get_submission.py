import pandas as pd
from utils import result2submission

result2submission(pd.read_csv("./data/bilstm-pred.csv"), "./data/bilstm-submission.csv")