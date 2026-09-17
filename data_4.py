import pandas as pd
import numpy as np

df = pd.read_csv("./data.csv")

df

df = df.drop('id', axis=1)

df


df.info()

print("Number of datasets:", df.shape)
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

df.head()


df['diagnosis'] = df['diagnosis'].map({
    'M': 1,
    'B': 0
})

df.head()

from sklearn.model_selection import train_test_split

x = df.drop('diagnosis', axis=1)
y = df['diagnosis']

x_train, x_test, y_train, y_test = train_test_split(x,y,
    test_size=0.2,
    random_state=42
)
     

print("Training Data Shape:", x_train.shape)
print("Testing Data Shape:", x_test.shape)

print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)