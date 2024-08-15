#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import Libraries


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#data


# In[10]:


train = pd.read_csv("C:/Users/yrsai/OneDrive/Desktop/intern/titanic_train.csv")
train.head(7)


# In[14]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='magma')


# In[18]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train, palette='coolwarm')


# In[62]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[57]:


sns.distplot(train['Age'].dropna(),kde=False,color='black',bins=40)


# In[55]:


train['Age'].hist(bins=30,color='black',alpha=0.3)


# In[23]:


sns.countplot(x='SibSp',data=train)


# In[50]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# In[49]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='summer')


# In[30]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[31]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[71]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='magma')


# In[33]:


train.drop('Cabin',axis=1,inplace=True)


# In[34]:


train.head()


# In[35]:


train.info()


# In[36]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.head()


# In[37]:


train = pd.concat([train,sex,embark],axis=1)
train.head()


# In[38]:


train.drop('Survived',axis=1).head()


# In[39]:


train['Survived'].head()


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# In[41]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[42]:


predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
accuracy=confusion_matrix(y_test,predictions)
accuracy


# In[43]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predictions)
accuracy


# In[44]:


predictions


# In[45]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[ ]:




