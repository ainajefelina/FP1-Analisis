# import pustaka
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# memuat dataset
df=pd.read_csv('uber.csv')

print(df.head())

# memilih variabel independen dan dependen
X =df[[ 'jarak','UberXL', 'Black', 'UberX','WAV', 'Black SUV', 'UberPool']]
y =df['harga']

# memisahkan dataset menjadi test dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=325)

#definisi model
lr = LinearRegression() 

# fit model
lr.fit(X_train, y_train)

#model pickle
pickle.dump(lr, open('model.pkl', 'wb'))