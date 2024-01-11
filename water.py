import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df=pd.read_csv('water_potability.csv')
df.head()

df.shape
df.fillna(df.mean(), inplace=True)

df.info()
df.describe()

#Data partition
X = df.drop('Potability',axis=1)
Y= df['Potability']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=101,shuffle=True)

Y_train.value_counts()
Y_test.value_counts()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
dt=DecisionTreeClassifier(criterion= 'gini', min_samples_split= 10, splitter= 'best')
dt.fit(X_train,Y_train)

pickle.dump(dt,open('water.pkl','wb'))



