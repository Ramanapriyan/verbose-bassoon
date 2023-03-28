#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt


def parser(x):
       return pd.datetime.strptime('190' +x, '%Y-%m')
 
series = pd.read_csv('D:\shampoo_sales.csv',header=0, parse_dates=[0], index_col=0,squeeze=True, date_parser=parser)

print(series.head())
series.plot()
plt.show()


# In[3]:


from pandas.plotting import autocorrelation_plot


autocorrelation_plot(series)
plt.show()


# In[4]:


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

#plot residuaal errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.title('ARMA Fit Residual Error Loine Plot')
plt.xlabel('Months')
plt.ylabel('Residual Error')
plt.show()


# In[5]:


residuals.plot(kind='kde')
plt.title('ARMA Fit Residual Error Density Plot')
plt.grid()
plt.show()
print(residuals.describe())


# In[16]:

# mean squared error
from sklearn.metrics import mean_squared_error

X = series.values
size = int(len(X) *0.66)


train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = []


for t in range(len(test)):
         model = ARIMA(history, order=(5,1,0))
         model_fit = model.fit(disp=0)
         output = model_fit.forecast()
         yhat = output[0]
         predictions.append(yhat)
         obs = test[t]
         history.append(obs)
         print('predicted=%f, expected=%f' % (yhat, obs))
            
            
            
            
#plot
plt.plot(test,label = 'original sales', marker = '*')
plt.plot(predictions, color='red', label = 'predicted sales', marker = '*')
plt.title('Performance Evaluation')
plt.xlabel('Future Steps')
plt.ylabel('Sales')
plt.legend()
plt.show()
            


# In[17]:


import math
error = mean_squared_error(test, predictions)
print('Test Root Mean Squared Error:%.3f' % math.sqrt(error))


# In[ ]:




