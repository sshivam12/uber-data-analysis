#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# 
# ## <b> Uber Data Analysis to predict the price

# In[ ]:


import pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import gc
import os
import sys
get_ipython().run_line_magic('matplotlib', 'inline')


# In[99]:


get_ipython().system('wget https://www.dropbox.com/s/ncqb2ctkg7da11k/weather.csv')


# In[100]:


get_ipython().system('wget https://www.dropbox.com/s/brixkogrmhan6ed/cab_rides.csv')


# In[101]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                #el
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[102]:


cab_data = pd.read_csv("/content/cab_rides.csv")
cab_data=reduce_mem_usage(cab_data)
weather_data = pd.read_csv("/content/weather.csv")
weather_data=reduce_mem_usage(weather_data)


# In[103]:


cab_data


# In[104]:


import datetime
cab_data['datetime']= pd.to_datetime(cab_data['time_stamp'])
cab_data
weather_data['date_time'] = pd.to_datetime(weather_data['time_stamp'])


# In[105]:


cab_data.columns


# In[106]:


weather_data.columns


# In[107]:


cab_data.shape


# In[108]:


weather_data.shape


# In[109]:


cab_data.describe()


# In[110]:


weather_data.describe()


# In[111]:


a=pd.concat([cab_data,weather_data])


# In[112]:


a['day']=a.date_time.dt.day
a['hour']=a.date_time.dt.hour


# In[113]:


a.fillna(0,inplace=True)


# In[114]:


a.columns


# In[115]:


a.groupby('cab_type').count()


# In[116]:


a.groupby('cab_type').count().plot.bar()


# In[117]:


a['price'].value_counts().plot(kind='bar',figsize=(100,50),color='blue')


# In[118]:


a['hour'].value_counts().plot(kind='bar',figsize=(10,5),color='blue')


# In[119]:


import matplotlib.pyplot as plt
x=a['hour']
y=a['price']
plt.plot(x,y)
plt.show()


# In[120]:


x=a['rain']
y=a['price']
plt.plot(x,y)
plt.show()


# In[121]:


a.columns


# In[122]:


x1=a[['distance', 'temp','clouds', 'pressure', 'humidity','wind','rain','day','hour','surge_multiplier','clouds']]
y1=a['price']


# In[123]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
x_train, y_train, x_test, y_test = train_test_split(x1, y1, test_size = 0.25, random_state = 42)


# In[124]:


linear=LinearRegression()
linear.fit(x_train,x_test)


# In[125]:


predictions=linear.predict(y_train)


# In[126]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
df


# In[127]:


df1 = df.head(25)
df1.plot(kind='bar',figsize=(26,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[127]:




