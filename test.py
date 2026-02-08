import pandas as pd
import random

df = pd.read_csv('data/processed/all_gpt4o_soap.csv')

# Show 5 random samples
for i in random.sample(range(len(df)), 5):
    row = df.iloc[i]
    print(f"\n{'='*60}")
    print(f"Sample {i+1} - {row['token_count']} tokens")
    print(f"{'='*60}")
    print(row['soap_note'])
    print()