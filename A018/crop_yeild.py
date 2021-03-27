#!/usr/bin/env python
# coding: utf-8

# In[124]:


#reading the imports

from numpy import array
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder


# In[71]:


#reading the data set
data_train = pd.read_csv('regressiondb.csv')

#reading the test data from the file
data_test = pd.read_excel('finalr.xlsx')


# In[220]:


print(data_train['Crop'].unique())


# In[219]:


#data_train


# In[218]:


#data_test


# In[76]:


label_encoder = LabelEncoder()


# In[77]:


temp = data_train['Crop']


# In[78]:


temp = array(temp)


# In[217]:


encoded = label_encoder.fit_transform(temp)
#print(encoded.tolist())


# In[81]:


temp1 = data_test['Crop']


# In[82]:


temp1 = array(temp1)


# In[83]:


encoded1 = label_encoder.transform(temp1)


# In[84]:


data_test['Crop'] = encoded1


# In[85]:


data_train['Crop'] = encoded


# In[214]:


#data_train


# In[213]:


#data_test


# In[91]:


x_train = data_train.iloc[:,0:4]


# In[93]:


x_train = array(x_train)


# In[94]:


y_train = data_train.iloc[:,4]


# In[95]:


y_train = array(y_train)


# In[96]:


x_test = data_test.iloc[:,0:4]


# In[97]:


y_test = data_test.iloc[:,4]


# In[98]:


x_test = array(x_test)


# In[99]:


y_test = array(y_test)


# In[101]:


'''regressor = LinearRegression()


# In[102]:


regressor.fit(x_train,y_train)


# In[103]:


y_pred = regressor.predict(x_test)'''


# In[105]:

regressor=DecisionTreeRegressor(random_state=0)
regressor.predict(X_train,y_train)

k = regressor.score(x_test,y_test)


# In[125]:


print("the accuracy is :",round(abs(k)*100,2 ),'%')


# In[186]:


data = [[400,40,9.7,7]]


# In[187]:


df = pd.DataFrame(data,columns = ['Rainfall','Temperature','Ph','Crop'])


# In[188]:


df


# In[189]:


x = df.iloc[:,:].values


# In[190]:


x


# model

# In[208]:


def pr(x):
    if(x[0][0]==0 and x[0][1]==0 and x[0][2]==0 and x[0][3]==0):
        return 0
        print("invalid s")
    elif(x[0][0]==0 or x[0][1]==0 or x[0][2] ==0 or x[0][3]==0):
        return 0
    else:
        print(":::The crop yielding based on the consider factors",regressor.predict(x),'in tonnes')
    


# In[209]:


pr(x)


# In[ ]:




