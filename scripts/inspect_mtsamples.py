"""
Dataset Inspection
"""

import pandas as pd

df = pd.read_csv('data/raw/mtsamples.csv')

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst sample transcription:")
print(df['transcription'].iloc[0][:500])
print("\nSpecialty distribution:")
print(df['medical_specialty'].value_counts().head(10))
print("\nTranscription length stats:")
print(df['transcription'].str.len().describe())