
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


dataframe = pd.read_csv("E:\housesalesprediction\kc_house_data.csv")


# In[5]:


dataframe.info()


# In[6]:


dataframe.describe()


# In[7]:


dataframe.isnull().sum()


# In[8]:


import seaborn as sns


# In[9]:


sns.pairplot(data = dataframe)


# In[15]:


features = dataframe[['bedrooms','bathrooms','sqft_living']]


# In[16]:


target = dataframe['id']


# In[34]:


import numpy as np
import matplotlib.pyplot as plt


# In[35]:


dataframe_normalized = np.log(target)


# In[36]:


plt.hist(dataframe_normalized)


# In[24]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train,X_test,Y_train,Y_test = train_test_split(features,target,test_size=0.2,train_size=0.8)


# In[27]:


from sklearn.metrics import r2_score


# In[28]:


from sklearn.linear_model import LinearRegression


# In[38]:


regressor = LinearRegression()
reg_fit = regressor.fit(X_train,Y_train)
reg_pred = reg_fit.predict(X_test)


# In[39]:


score_not_norm = r2_score(Y_test,reg_pred)


# In[40]:


print(score_not_norm)


# In[41]:


X_train_n,X_test_n,Y_train_n,Y_test_n = train_test_split(features,dataframe_normalized,test_size=0.2,train_size=0.8)


# In[42]:


regressor_norm = LinearRegression()
reg_fit_norm = regressor.fit(X_train_n,Y_train_n)
reg_pred_norm = reg_fit.predict(X_test_n)


# In[43]:


score_norm = r2_score(Y_test_n,reg_pred_norm)


# In[44]:


print(score_norm)

