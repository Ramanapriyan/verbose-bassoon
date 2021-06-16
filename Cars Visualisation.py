#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


# In[6]:


df = pd.read_csv('D:\CARS.csv')
df.head(5)


# In[6]:


df = df.drop(['Model','DriveTrain','Invoice', 'Origin', 'Type'], axis=1)
df.head(5)


# In[7]:


df.info()


# In[8]:


df.shape


# In[9]:


df = df.drop_duplicates(subset='MSRP', keep='first')
df.count()


# In[10]:


print(df.isnull().sum())


# In[9]:


df[247:249]


# In[13]:


# Filling the rows with the mean of the column

val = df['Cylinders'].mean()
df['Cylinders'][247] = round(val)
val = df['Cylinders'].mean()
df['Cylinders'][248]= round(val)


# In[14]:


# Removing the formatting
df['MSRP'] = [x.replace('$', '') for x in df['MSRP']] 
df['MSRP'] = [x.replace(',', '') for x in df['MSRP']]
df['MSRP']=pd.to_numeric(df['MSRP'],errors='coerce')


# In[15]:


sns.boxplot(x=df['MSRP'])


# In[17]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[21]:


df = df[~((df < (Q1-1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[20]:


sns.boxplot(x=df['MSRP'])


# In[22]:


df.describe()


# In[24]:


# Plotting a heat map
plt.figure(figsize=(10,6))
c= df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)


# In[25]:


# Plotting a scatter plot
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(df['Horsepower'], df['MSRP'])
plt.title('Scatter plot between MSRP and Horsepower')
ax.set_xlabel('Horsepower')
ax.set_ylabel('MSRP')
plt.show()


# In[ ]:




