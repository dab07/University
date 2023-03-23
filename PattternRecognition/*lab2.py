import pandas as pd
import numpy as np

dataset = pd.read_csv("/Users/hs/Downloads/tennis.csv")
print(dataset.head() + "\n")
print(dataset.describe() + "\n\n")

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1:]

print(x)
print(y)