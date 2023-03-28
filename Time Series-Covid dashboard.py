#!/usr/bin/env python
# coding: utf-8

# In[1]:

# importing libraries:
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import itertools
import warnings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# In[2]:


df = pd.read_csv('D:\covid_19_india.csv')
df.head(5)


# In[7]:


df.isnull().sum()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


print(df['Deaths'].unique())


# In[11]:


df['Deaths'].astype('int64')


# In[12]:


df['State/UnionTerritory'].unique()


# In[13]:


def drop_star(df):
    for i in df['State/UnionTerritory'].iteritems():
        if i[1][-3:] == "***":
            df.drop(i[0],inplace=True)
            
drop_star(df)
df['State/UnionTerritory'].unique()


# In[14]:


df['Datetime'] = df['Date']+' '+df['Time']


# In[15]:


l = df.groupby('State/UnionTerritory')
current = l.last()


# In[16]:


fig ,ax = plt.subplots(figsize= (12,8))
plt.title('Top 10 Contaminated States')
current = current.sort_values("Confirmed",ascending=False)[:10]
p = sns.barplot(ax=ax,x= current.index,y=current['Confirmed'])
p.set_xticklabels(labels = current.index,rotation=90)
p.set_yticklabels(labels=(p.get_yticks()*1).astype(int))
plt.show()


# In[17]:


df.tail()


# In[24]:


l = df.groupby('State/UnionTerritory')
current = l.last()
current = current.sort_values("Confirmed",ascending=False)


# In[19]:


df['Date'].min(), df['Date'].max()


# In[31]:


TN = df.loc[df['State/UnionTerritory'] == 'Tamil Nadu']
TN.head()


# In[21]:


TN1=TN[:10]


# In[22]:


TN1


# In[31]:


plt.pie(
    TN1['Confirmed'],
    labels=TN1['State/UnionTerritory'])


# In[27]:


TN.shape


# In[28]:


TN.isnull().sum()


# In[29]:


TN.columns


# In[35]:


cols=['Sno','Time','State/UnionTerritory','ConfirmedIndianNational','ConfirmedForeignNational','Cured','Deaths']
TN['Date'] = TN['Date']+' '+TN['Time']
TN.drop(cols, axis=1, inplace=True)
TN= TN.sort_values('Date')
TN.isnull().sum()


# In[36]:


TN.head()


# In[37]:


TN.index


# In[39]:


TN = TN.groupby('Date')['Confirmed'].sum().reset_index()


# In[40]:


TN = TN.set_index('Date')
TN.index = pd.to_datetime(TN.index)
TN.index


# In[41]:


y = TN['Confirmed'].resample('W').mean()


# In[42]:


y.index


# In[43]:


y.fillna(method='ffill',inplace=True)
y['2020':]


# In[46]:


TN.plot(figsize=(16,6))
plt.show()


# In[47]:


from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, freq = 20, model='additive')
fig = decomposition.plot()
plt.show()


# In[49]:


p = d = q =range(0,2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d,q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[51]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationary=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}7 - AIC:{}'.format(param, param_seasonal,results.aic))
        except:
            continue


# In[52]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0,1,1),
                                seasonal_order=(1,1,1,12),
                                enforce_stationary=False,
                                enforce_invertibility=False)
results = mod.fit()


# In[55]:


pred = results.get_prediction(start=pd.to_datetime('2020-08-02'),dynamic=False)
pred_ci = pred.conf_int()
ax = y['2020':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast',alpha=.7, figsize=(14,7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Confirmed Cases')
plt.legend()
plt.show()


# In[56]:


pred_uc = results.get_forecast(steps=50)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Confirmed Cases')
plt.legend()
plt.show()


# In[ ]:




