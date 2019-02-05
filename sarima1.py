
# coding: utf-8

# In[2]:


import os
os.chdir('C:\\Users\\vyasy\\Desktop\\Data')


# In[3]:


from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.api as sm


# In[4]:


def parse(x):
    return datetime.strptime(x,'%Y-%m')


# In[5]:


dataset = read_csv('adv_sales.csv',index_col = 0 , date_parser = parse)


# In[6]:


pyplot.xticks(rotation=70)
pyplot.grid()
pyplot.plot(dataset)
pyplot.show()


# In[7]:


series = dataset['Sales']


# In[8]:


decomposition = seasonal_decompose(series, freq=12)  
fig = pyplot.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)


# In[9]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    fig = pyplot.figure(figsize=(12, 8))
    orig = pyplot.plot(timeseries, color='blue',label='Original')
    mean = pyplot.plot(rolmean, color='red', label='Rolling Mean')
    std = pyplot.plot(rolstd, color='black', label = 'Rolling Std')
    pyplot.legend(loc='best')
    pyplot.title('Rolling Mean & Standard Deviation')
    pyplot.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# In[10]:


test_stationarity(series)


# In[11]:


first_difference = series - series.shift(1)


# In[12]:


test_stationarity(first_difference.dropna(inplace = False))


# In[13]:


plot_acf(first_difference[13:])
plot_pacf(first_difference[13:])
pyplot.grid()


# In[14]:


mod = sm.tsa.statespace.SARIMAX(series, trend='n', order=(0,1,0), seasonal_order=(1,1,0,12),enforce_stationarity=False,enforce_invertibility=False)
results = mod.fit()
print(results.summary())


# In[15]:


dataset['forcast'] = results.predict(start = 24, end= 36,interval = 'confidence',level = 0.95)  
dataset[['Sales', 'forcast']].plot(figsize=(12, 8))
pred_ci = results.conf_int()


# In[19]:


data1 = results.predict(start = 24, end = 35)


# In[20]:


data2 = series[24:]


# In[21]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(data2,data1))


# In[22]:


rms

