#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas seaborn numpy scipy torch torchvision tqdm sklearn ')


# In[2]:


import pandas as pd
import seaborn as sns; sns.set()
import numpy as np
import re

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 8

from scipy.stats import norm
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn import neighbors

import warnings
import seaborn as sns

import math
import decimal

warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('dataset_mood_smartphone.csv')


# In[4]:


df.columns


# In[5]:


df.head(30)


# In[6]:


df


# In[7]:


df['id'].describe


# In[8]:


df['id'].value_counts()


# In[9]:


df['variable'].describe


# In[10]:


df['variable'].value_counts()


# In[11]:


df['value'].describe


# In[12]:


df['value'].value_counts()


# In[13]:


df.info()


# In[14]:


print("id:\n", set(df['id']),
     "\n\nvariable:\n", set(df['variable']))


# In[15]:


sns.countplot(df['variable'], data=df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


df.isnull().sum()


# In[17]:


df.columns


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


df


# In[19]:


df = df.drop(columns=(['Unnamed: 0']))


# In[20]:


df


# In[ ]:





# In[21]:


df


# In[22]:


df['time'] = df['time'].astype('datetime64').dt.date


# In[23]:


df


# In[24]:


df.groupby('time')


# In[25]:


df


# In[26]:


def compute_avg_val(df):
    df['daily average'] = df['value'].mean()
    return df
grouped = df.groupby(['time', 'id','variable'])
df = grouped.apply(compute_avg_val)


# In[27]:


df


# In[28]:


df = df.drop_duplicates(subset=['time', 'id','variable'])


# In[29]:


df


# In[30]:


df.head(50)


# In[31]:


df = df.drop(columns=(['value']))


# In[32]:


df.loc[(df['id'] == 'AS14.01') & (df['time'].astype('datetime64') == '2014-02-26')]


# In[33]:


df


# In[34]:


#Add mood
df3 = df.loc[df['variable'] == 'mood']
df['mood']=df3['daily average'].astype(float) 


# In[35]:


#Add circumplex.arousal
df3 = df.loc[df['variable'] == 'circumplex.arousal']
df['circumplex.arousal']=df3['daily average'].astype(float) 


# In[36]:


#Add circumplex.valence
df3 = df.loc[df['variable'] == 'circumplex.valence']
df['circumplex.valence']=df3['daily average'].astype(float) 


# In[37]:


#Add activity
df3 = df.loc[df['variable'] == 'activity']
df['activity']=df3['daily average'].astype(float) 


# In[38]:


#Add screen
df3 = df.loc[df['variable'] == 'screen']
df['screen']=df3['daily average'].astype(float) 


# In[39]:


#Add call
df3 = df.loc[df['variable'] == 'call']
df['call']=df3['daily average'].astype(float) 


# In[40]:


#Add sms
df3 = df.loc[df['variable'] == 'sms']
df['sms']=df3['daily average'].astype(float) 


# In[41]:


#Add appCat.builtin
df3 = df.loc[df['variable'] == 'appCat.builtin']
df['appCat.builtin']=df3['daily average'].astype(float) 


# In[42]:


#Add appCat.communication
df3 = df.loc[df['variable'] == 'appCat.communication']
df['appCat.communication']=df3['daily average'].astype(float) 


# In[43]:


#Add appCat.entertainment
df3 = df.loc[df['variable'] == 'appCat.entertainment']
df['appCat.entertainment']=df3['daily average'].astype(float) 


# In[44]:


#Add appCat.finance
df3 = df.loc[df['variable'] == 'appCat.finance']
df['appCat.finance']=df3['daily average'].astype(float) 


# In[45]:


#Add appCat.game
df3 = df.loc[df['variable'] == 'appCat.game']
df['appCat.game']=df3['daily average'].astype(float) 


# In[46]:


#Add appCat.office
df3 = df.loc[df['variable'] == 'appCat.office']
df['appCat.office']=df3['daily average'].astype(float) 


# In[47]:


#Add appCat.other
df3 = df.loc[df['variable'] == 'appCat.other']
df['appCat.other']=df3['daily average'].astype(float) 


# In[48]:


#Add appCat.social
df3 = df.loc[df['variable'] == 'appCat.social']
df['appCat.social']=df3['daily average'].astype(float) 


# In[49]:


#Add appCat.travel
df3 = df.loc[df['variable'] == 'appCat.travel']
df['appCat.travel']=df3['daily average'].astype(float) 


# In[50]:


#Add appCat.unknown
df3 = df.loc[df['variable'] == 'appCat.unknown']
df['appCat.unknown']=df3['daily average'].astype(float) 


# In[51]:


#Add appCat.utilities
df3 = df.loc[df['variable'] == 'appCat.utilities']
df['appCat.utilities']=df3['daily average'].astype(float) 


# In[52]:


#Add appCat.weather
df3 = df.loc[df['variable'] == 'appCat.weather']
df['appCat.weather']=df3['daily average'].astype(float) 


# In[53]:


df


# In[54]:


df3


# In[55]:


df_grouped = df.groupby(['time', 'id'])


# In[56]:


df_grouped.head(30)


# In[57]:


df.head(30)


# In[58]:


df['time'] = pd.to_datetime(df['time'])


# In[59]:


df = df.drop(columns=(['variable', 'daily average']))


# In[60]:


df


# In[61]:


#drop rows with more than 0 Nan values
df_new = df.loc[df.isnull().sum(axis=1)<1]


# In[62]:


df_new


# In[63]:


df


# In[64]:


#group by sample number
df_grouped = df.groupby((['id', 'time']), as_index=False).max()


# In[65]:


#drop rows with more than 18 Nan values
df_new = df_grouped.loc[df_grouped.isnull().sum(axis=1)<18]


# In[66]:


df_new


# In[67]:


df_new.loc[df_new['id'] == 'AS14.01']


# In[68]:


cols = df_new.columns
corrmat = df_new[cols].corr()
fig, ax = plt.subplots(figsize=(19,19)) 
sns.heatmap(corrmat, square=True, annot=True, ax = ax)


# In[69]:


plt.scatter(df_new['mood'],df_new['circumplex.valence'], color='red', linewidth=2)


# In[70]:


plt.scatter(df_new.loc[df_new['id'] == 'AS14.01', 'time'],df_new.loc[df_new['id'] == 'AS14.01', 'mood'], color='red', linewidth=2)


# In[71]:


df_new = df_new.reset_index().drop(columns=(['index']))


# In[ ]:





# In[72]:


df_new


# In[ ]:




