#!/usr/bin/env python
# coding: utf-8

# ## Upload and Explore dataset

# In[2]:


import pandas as pd


# In[3]:


#Loading the dataset

df = pd.read_csv("Churn_Modelling.csv")

df.info()

df.head()


# In[4]:


#any duplicated

df.duplicated().sum() 


# In[5]:


df2 = df.copy()


# In[6]:


#dropping unnessecary column

df2= df2.drop(['Surname', 'CustomerId', 'RowNumber'], axis=1, errors='ignore')
df2.info()


# In[7]:


df2['Gender'].value_counts()


# In[8]:


df2['Geography'].value_counts()


# In[9]:


columns = list(df2.columns)

categoric_columns = []
numeric_columns = []

for i in columns:
    if len(df[i].unique()) > 6:
        numeric_columns.append(i)
    else:
        categoric_columns.append(i)

categoric_columns = categoric_columns[:-1] # Excluding target:'Exited'


# In[10]:


print('Numerical fetures: ',numeric_columns)
print('Categorical fetures: ',categoric_columns)


# In[11]:


X = df2.drop('Exited', axis=1)
y = df2['Exited']


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.20, random_state = 42)


# In[14]:


preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),                      
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categoric_columns),  
    ],
    remainder='drop'  
)

Xtr = preprocess.fit_transform(X_train)   # fit sadece train
Xte = preprocess.transform(X_test)


# In[15]:


Xtr.shape, Xte.shape


# ## ANN

# In[16]:


from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score


# In[17]:


model = Sequential([
     Input(shape=(Xtr.shape[1],)),      #
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1,  activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[18]:


model.fit(Xtr, y_train, validation_split=0.2, epochs=20)


# In[19]:


y_pred = (model.predict(Xte, verbose=0) >= 0.5).astype(int)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[ ]:




