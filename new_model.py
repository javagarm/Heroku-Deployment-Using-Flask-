#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import matplotlib.pyplot as plt
import seaborn
import pandas as pd

ticker = yf.Ticker("GC=F")


# get historical market data
df = ticker.history(period="300d")

df=df.reset_index()


# In[2]:


df=df.drop(['Open','Close','Low','Volume','Dividends','Stock Splits'],axis=1)


# In[3]:


x=df.drop('High',axis=1)

y=df[['High']]


# In[4]:


x['Date']=pd.to_datetime(df['Date']).values.astype(int)


# In[5]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state =100)


# In[6]:


#random forest regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 1) 
rf.fit(x_train,y_train)


# In[7]:


# linear regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)


# In[10]:


#decision tree regression
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=6,criterion="mse",random_state=0)
dt.fit(x_train,y_train)


# In[9]:


import pickle
pickle.dump(rf, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

