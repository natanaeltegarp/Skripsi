import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('trainSet-stelma.csv')

train_df = pd.DataFrame()
test_df = pd.DataFrame()

for idpsj, group in df.groupby('IDPSJ'):
    train, test = train_test_split(group, test_size=0.2, random_state=42, shuffle=True)
    train_df = pd.concat([train_df, train], ignore_index=True)
    test_df = pd.concat([test_df, test], ignore_index=True)

train_df.to_csv('stelma-train-split.csv', index=False)
test_df.to_csv('stelma-test-split.csv', index=False)

print("Split success")
