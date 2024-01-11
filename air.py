import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


df = pd.read_csv('train.csv')
df.fillna(df.mean(), inplace=True)
df.head()


x_train = df.drop(['target'],axis= 1)
y_train = df['target']

model = LinearRegression()
model.fit(x_train,y_train)

pickle.dump(model,open('air.pkl','wb'))
