import pandas as pd
df = pd.read_csv("data/raw/mtsamples.csv")
print(df.describe())