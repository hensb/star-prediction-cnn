import pandas as pd
import numpy as np

df = pd.read_csv('./resources/Reviews.csv')
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.8

train = df[msk]
test = df[~msk]

train.to_csv('output/reviews.train.csv', index=False)
test.to_csv('output/reviews.test.csv', index=False)