
# coding: utf-8

# In[17]:


from random import seed
from random import random
from matplotlib import pyplot as plt
import numpy as np
from pandas.tools.plotting import autocorrelation_plot
from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import pandas as pd


# In[54]:


sales = [15,16,18,27,21,49,21,22,28,36,40,3,21,29,62,65,46,44,33,62,22,12,24,3,5,14,36,40,49,7,52,65,17,5,17,1]
time = np.arange(1,37,1)
time1  = [0] * 23
dataset = pd.DataFrame({'sales':sales,'time':time})


# In[55]:


plt.plot(time,sales)
plt.grid()
plt.show


# In[4]:


plot_acf(sales)
plt.grid()


# In[5]:


plot_pacf(sales)
plt.grid()


# In[6]:


## ARMA model (1,1) AIC BIC


# In[7]:


arma_mod = sm.tsa.ARMA(sales, order=(1,1))
arma_res = arma_mod.fit(trend='nc', disp=-1)


# In[8]:


print(arma_res.summary())


# In[50]:


forecast0 =  arma_res.predict(start= 25 , end =  36)
forecast0


# In[42]:





# In[43]:


from sklearn.metrics import mean_squared_error
from math import sqrt


# In[51]:


data1 = [5,14,36,40,49,7,52,65,17,5,17,1]
data2 = forecast0


# In[52]:


rms = sqrt(mean_squared_error(data1,data2))


# In[53]:


rms


# In[56]:


data1


# In[57]:


forecast0

