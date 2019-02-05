
# coding: utf-8

# In[1]:


import os
os.chdir('C:\\Users\\vyasy\\Desktop\\Data')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 15, 20
from pandas import read_csv
from datetime import datetime


# In[4]:


def parse(x):
    return datetime.strptime(x,'%Y-%m')


# In[5]:


dataset = read_csv('adv_sales.csv',index_col = 0 , date_parser = parse)


# In[6]:


series = dataset['Sales']


# In[7]:


series


# In[8]:


plt.plot(series)


# In[9]:


p = d = q = range(0,2)


# In[10]:


pdq = list(itertools.product(p,d,q))


# In[11]:


pdq


# In[12]:


seasonal_pdq = [(x[0],x[1],x[2],12)for x in list(itertools.product(p,d,q))]


# In[13]:


seasonal_pdq


# In[14]:


warnings.filterwarnings("ignore")
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(series,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,
                                                                                        enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[15]:


mod1 = sm.tsa.statespace.SARIMAX(series, order = (1,1,0),seasonal_order = (1,1,0,12),enforce_stationarity=False,enforce_invertibility=False)


# In[16]:


results = mod1.fit()


# In[17]:


print(results.summary().tables[1])


# In[19]:


pred = results.get_prediction(start=pd.to_datetime('2003-01-01'), dynamic=False)
pred_ci = pred.conf_int()

ax = series['2001':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()
plt.grid()
plt.show()

