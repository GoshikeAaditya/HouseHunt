import pandas as pd

# Update the filename if your CSV has a different name
df = pd.read_csv("//Users/aaditya/Documents/househunt/dataset_indian.csv")

# Show the first few rows
print("First 5 rows:")
print(df.head())

# Show basic info
print("\nDataset Info:")
print(df.info())

# Show summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Show column names
print("\nColumns:")
print(df.columns)