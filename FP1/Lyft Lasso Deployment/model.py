# import pustaka
import pandas as pd
from pandas import read_csv
from numpy import arange
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
import pickle

# memuat dataset
df=pd.read_csv('lyft.csv')

print(df.head())

# memilih variabel independen dan dependen
X =df[[ 'jarak','lonjakan','Shared','Lux','Lyft','Lux Black XL','Lyft XL','Lux Black']]
y =df['harga']

# memisahkan dataset menjadi test dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=325)

#cv
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

#definisi model
lasso= LassoCV(cv=cv)

# fit model
lasso.fit(X_train, y_train)

#model pickle
pickle.dump(lasso, open('model.pkl', 'wb'))