
# coding: utf-8

# In[295]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import seaborn as sns
# import talib
import warnings
warnings.filterwarnings('ignore')
import datetime
pd.set_option('display.max_columns',36)


# In[296]:


dataset = pd.read_csv('TITAN.NS (2).csv')


# In[297]:


df = dataset


# In[298]:


dataset['H-L'] = dataset['High'] - dataset['Low']
dataset['O-C'] = dataset['Close'] - dataset['Open']
dataset['3day MA'] = dataset['Close'].shift(1).rolling(window = 3).mean()
dataset['10day MA'] = dataset['Close'].shift(1).rolling(window = 10).mean()
dataset['30day MA'] = dataset['Close'].shift(1).rolling(window = 30).mean()
dataset['Std_dev']= dataset['Close'].rolling(5).std()


# # CALCULATING RSI

# In[299]:


dataset['Change'] = dataset['Close']-dataset['Close'].shift(1)
# dataset.head()


# In[300]:


dataset['Change'].head()


# In[301]:


dataset['Upward Movement'] = np.where(dataset['Change']>0,dataset['Change'],0)
dataset['Downward Movement'] = np.where(dataset['Change']<0,abs(dataset['Change']),0)
#for CALCULATING RSI


# In[302]:


dataset.head()


# In[303]:


dataset['Average Upward Movement'] = dataset['Upward Movement'].shift(1).rolling(window = 10).mean()
dataset['Average Downward Movement'] = dataset['Downward Movement'].shift(1).rolling(window = 10).mean()


# In[304]:


dataset.head(0)


# In[305]:


dataset['Average Upward Movement'] = (dataset['Average Upward Movement'].shift(-1)*9+dataset['Upward Movement'])/10
#The first row is an average and following rows are exponential moving averages


# In[306]:


dataset['Average Downward Movement'] = (dataset['Average Downward Movement'].shift(-1)*9+dataset['Downward Movement'])/10
#The first row is an average and following rows are exponential moving averages


# In[307]:


dataset['Relative Strength'] = dataset['Average Upward Movement']/dataset['Average Downward Movement']


# In[308]:


dataset['RSI'] = 100-100/(dataset['Relative Strength']+1)


# # calculate william r%

# In[309]:


dataset['Max high'] = dataset['High'].shift(1).rolling(window = 10).max()


# In[310]:


dataset['Min low']  = dataset['Low'].shift(1).rolling(window = 10).min()


# In[311]:


dataset['William %R'] = ((dataset['Max high']-dataset['Close'])/(dataset['Max high']-dataset['Min low']))*(-100)


# DROPPING NAN VALUES 

# In[312]:


dataset = dataset.dropna()


# In[313]:


dataset.head(1)


# In[314]:


#adding new features 

def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    smas = np.convolve(values, weights, 'valid')
    return smas

def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1.0,0.,window))
    weights /= weights.sum()

    a = np.convolve(values, weights)[:len(values)]
    a[:window] = a[window]
    return a 


# In[315]:


dataset['expFast'] = ExpMovingAverage(dataset['Close'], 10)
dataset['expSlow'] = ExpMovingAverage(dataset['Close'], 30)
dataset['MACD'] = dataset['expFast'] - dataset['expSlow']#Moving Average Convergence Divergence


# In[316]:


dataset.head(0)


# In[317]:


dataset['Price Rise'] = np.where(dataset['Close'].shift(-1) > dataset['Close'], 1, 0)


# In[318]:


# data_features = dataset[['H-L','O-C','3day MA','10day MA','30day MA','Std_dev','RSI','William %R','expFast','expSlow','MACD','Price Rise']]


# In[319]:


data_features = dataset[['H-L','O-C','10day MA','Std_dev','RSI','William %R','expFast','expSlow','MACD','Price Rise']]


# In[320]:


data_features.reset_index(inplace = True)


# In[321]:


data_features.drop('index',axis = 1, inplace = True)
data_features.head()


ANALYZING OUTLIERS

# In[322]:


fig, ax = plt.subplots(figsize=(8,  8))
sns.boxplot(['H-L','O-C','3day MA','10day MA','30day MA','Std_dev','RSI','William %R'])

plt.figure(figsize=(20,4))

plt.subplot(1,8,1)
sns.boxplot(y = data_features['H-L'])
plt.title('H-L')

plt.subplot(1,8,2)
sns.boxplot(y = data_features['O-C'])
plt.title('O-C')

plt.subplot(1,8,3)
sns.boxplot(y = data_features['3day MA'])
plt.title('3day MA')

plt.subplot(1,8,4)
sns.boxplot(y = data_features['10day MA'])
plt.title('10day MA')

plt.subplot(1,8,5)
sns.boxplot(y = data_features['30day MA'])
plt.title('30day MA')

plt.subplot(1,8,6)
sns.boxplot(y = data_features['Std_dev'])
plt.title('Std dev')

plt.subplot(1,8,7)
sns.boxplot(y = data_features['RSI'],color = 'r')
plt.title('RSI')

plt.subplot(1,8,8)
sns.boxplot(y = data_features['William %R'], color = 'r')
plt.title('William %R')


# In[323]:


X = data_features.iloc[:,:-1]
y = data_features.iloc[:,-1]

split = int(len(dataset)*0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]


# In[324]:


y_test.unique()


# In[325]:



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[326]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[327]:


classifier = Sequential()


# In[328]:


classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))#ouput layer


# In[329]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[330]:


classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# In[331]:


y_pred = classifier.predict(X_test)


# In[332]:


y_pred = (y_pred > 0.5)


# In[333]:


from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test)


# In[284]:


data_features.head(5)


# In[285]:



dataset['y_pred'] = np.NaN
dataset.iloc[(len(dataset) - len(y_pred)):,-1:] = y_pred
trade_dataset = dataset.dropna()#store dataset values to new dataset and dropping NaN values


# In[286]:




trade_dataset['Returns'] = 0.
trade_dataset['Returns'] = np.log(trade_dataset['Close']/trade_dataset['Close'].shift(1))
trade_dataset['Returns'] = trade_dataset['Returns'].shift(-1)


# In[290]:



trade_dataset['Predicted Returns'] = 0.
trade_dataset['Predicted Returns'] = np.where(trade_dataset['y_pred'] == True, trade_dataset['Returns'], - trade_dataset['Returns'])


# In[291]:


trade_dataset['Cumulative Market Returns'] = np.cumsum(trade_dataset['Returns'])
trade_dataset['Cumulative Predicted Returns'] = np.cumsum(trade_dataset['Predicted Returns'])


# In[292]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(trade_dataset['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(trade_dataset['Cumulative Predicted Returns'], color='g', label='Predicted Returns')
plt.grid()
plt.legend()
plt.savefig('sample.png')

plt.show()


