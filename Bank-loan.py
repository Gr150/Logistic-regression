#!/usr/bin/env python
# coding: utf-8

# In[13]:


#import the libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression


# In[14]:


#Load the dataset
client=pd.read_csv("bank-full.csv")
client.head()
client.info()


# In[15]:


#Output y -> Whether the client has subscribed a term deposit or not Binomial ("yes" or "no")
client.loc[:,"default"]


# In[16]:


client.head()


# In[29]:


#There are categorical variables and they need to converted into numerical value or else the logisctic regression will throw an error
#label encoding to be used when converting Y variable

#one hot encoding for converting x variable date

client=pd.get_dummies(client.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]])


# In[ ]:





# In[30]:


#Divide data into input and output variable (X, Y)
X=client.iloc[:,0:15,]
Y=client.iloc[:,16]


# In[31]:


classifier = LogisticRegression()
classifier.fit(X,Y)


# In[20]:


#Predict using X variable
y_predit=classifier.predict(x)


# In[ ]:




