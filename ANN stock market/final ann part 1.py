#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import talib


# In[ ]:


import random
random.seed(42)#TO START WITH SAME SEED EVERY TIME


# In[ ]:


dataset = pd.read_csv("/home/soumik/Documents/i3 financial/TITAN.NS.csv")
dataset = dataset.dropna()
dataset = dataset[['Open', 'High', 'Low', 'Close']]


# In[ ]:


#PREPARING THE DATASET
dataset['H-L'] = dataset['High'] - dataset['Low']
dataset['O-C'] = dataset['Close'] - dataset['Open']
dataset['3day MA'] = dataset['Close'].shift(1).rolling(window = 3).mean()
dataset['10day MA'] = dataset['Close'].shift(1).rolling(window = 10).mean()
dataset['30day MA'] = dataset['Close'].shift(1).rolling(window = 30).mean()
dataset['Std_dev']= dataset['Close'].rolling(5).std()
dataset['RSI'] = talib.RSI(dataset['Close'].values, timeperiod = 9)
dataset['Williams %R'] = talib.WILLR(dataset['High'].values, dataset['Low'].values, dataset['Close'].values, 7)


# In[ ]:


dataset['Price_Rise'] = np.where(dataset['Close'].shift(-1) > dataset['Close'], 1, 0)
#IT STORES 1 WHEN TOMMOROW CLOSING PRICE IS GREATER  THAN TODAYS


# In[ ]:


dataset = dataset.dropna()


# In[ ]:


dataset.head()


# In[ ]:


X = dataset.iloc[:, 4:-1]
y = dataset.iloc[:, -1]


# In[ ]:


split = int(len(dataset)*0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[ ]:


classifier = Sequential()


# In[ ]:


classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))


# In[ ]:


classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))#ouput layer


# In[ ]:


classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])


# In[ ]:


classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# In[ ]:


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)#to store binary values greater or less


# In[ ]:


dataset['y_pred'] = np.NaN
dataset.iloc[(len(dataset) - len(y_pred)):,-1:] = y_pred
trade_dataset = dataset.dropna()#store dataset values to new dataset and dropping NaN values


# In[ ]:


trade_dataset['Tomorrows Returns'] = 0.
trade_dataset['Tomorrows Returns'] = np.log(trade_dataset['Close']/trade_dataset['Close'].shift(1))
trade_dataset['Tomorrows Returns'] = trade_dataset['Tomorrows Returns'].shift(-1)


# In[ ]:


trade_dataset['Strategy Returns'] = 0.
trade_dataset['Strategy Returns'] = np.where(trade_dataset['y_pred'] == True, trade_dataset['Tomorrows Returns'], - trade_dataset['Tomorrows Returns'])


# In[ ]:


trade_dataset['Cumulative Market Returns'] = np.cumsum(trade_dataset['Tomorrows Returns'])
trade_dataset['Cumulative Strategy Returns'] = np.cumsum(trade_dataset['Strategy Returns'])


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(trade_dataset['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(trade_dataset['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
plt.legend()
plt.show()


# In[ ]:
#accuracy = 92.44%
