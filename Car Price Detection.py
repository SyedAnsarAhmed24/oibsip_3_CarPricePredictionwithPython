#!/usr/bin/env python
# coding: utf-8

# # OASIS INFOBYTE SIP August data science Internship TASK 3
# 
# 
# 

# # Car Price Prediction Using Machine Learning.

# # IMPORT LIBRARIES

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
print("libraries imported")


# # Importing Dataset
# 

# In[3]:


df=pd.read_csv("C:\\Users\\SYED ANSAR AHMED\\OneDrive\\Desktop\\CarPrice_Assignment.csv")
df.head()


# In[3]:


df.tail()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[4]:


sns.set_style('whitegrid')
plt.figure(figsize=(15,7))
sns.histplot(df.price,kde=True,color='grey')
plt.show()


# In[8]:


df.corr()


# # visualizating the dataset

# In[5]:


plt.figure(figsize = (15, 15))

corr= df.corr()

cmap= matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["#00aabb", "#ffffff", "#ff7777"])

sns.heatmap(corr, cmap = cmap, annot=True)

plt.show()


# In[4]:


plt.figure(figsize = (30,30))
sns.pairplot(df)
plt.show()


# In[5]:


plt.figure(figsize=(20, 12))

plt.subplot(3, 3, 1) # 3 rows, 3 columns, 1st subplot = left
sns.boxplot(x='fueltype', y='price', data=df)

plt.subplot(3, 3, 2) # 3 rows, 3 columns, 2nd subplot = middle
sns.boxplot(x='aspiration', y='price', data=df)

plt.subplot(3, 3, 3) # 3 rows, 3 columns, 3rd subplot = right
sns.boxplot(x='carbody', y='price', data=df)

plt.subplot(3, 3, 4)
sns.boxplot(x='drivewheel', y='price', data=df)

plt.subplot(3, 3, 5)
sns.boxplot(x='enginelocation', y='price', data=df)

plt.subplot(3, 3, 6)
sns.boxplot(x='enginetype', y='price', data=df)

plt.subplot(3, 3, 7)
sns.boxplot(x='fuelsystem', y='price', data=df)


# In[6]:


plt.figure(figsize=(20, 10))
sns.boxplot(x='CarName', y='price', data=df)


# # Data Preparation Sepereating Input Columns And The Output Column

# In[8]:


df=df[["symboling","wheelbase","carwidth","carheight","curbweight","enginesize","stroke","compressionratio","horsepower",
       "peakrpm","citympg","highwaympg","price"]]
x=np.array(df.drop("price",axis=1))
y=np.array(df["price"])
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


# # Scaling out data

# In[9]:


scaler=MinMaxScaler()
xtrain=scaler.fit_transform(xtrain)
xtest=scaler.transform(xtest)


# # Training a Car Price Prediction Model

# In[10]:


model=DecisionTreeRegressor()
model.fit(xtrain,ytrain)
prediction=model.predict(xtest)


# # Evaluate the model

# In[11]:


print(f"Accuracy:{model.score(xtest,prediction)*100}")
print(f"R Squared Error:{round(metrics.r2_score(ytest,prediction)*100,2)}")
print(f"Absolute Error:{metrics.mean_absolute_error(ytest,prediction)}")
print(f"Mean Squared Error:metrics.mean_squared_error(ytest,prediction)")
print(f"Root Squarred Error:mertices.root_squared_error(ytest,prediction)")


# # Plotting Actaul and Predicted price

# In[12]:


plt.scatter(ytest,prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price ")
plt.title("Actual Price vs Predicted Price")
plt.show()


# # Testing our predction with some other example data

# In[13]:


def prediction(input_data):
   #input data
    car_data=df.drop('price',axis=1).iloc[input_data]
    #scaling input data
    car_scaled = scaler.transform(car_data.values.reshape(1, -1))
    predicted_price = model.predict(car_scaled)

    #predict price of input data 
    print(f"Predicted Price of this car is :{predicted_price[0]}")
    print()
    #actual price
    real_price=df.iloc[input_data].price
    print(f"Real Price of car is : {real_price}")


# In[14]:


prediction(12)


# In[15]:


prediction(2)


# In[16]:


prediction(9)


# In[17]:


prediction(5)


# In[18]:


prediction(7)


# In[19]:


prediction(8)


# In[ ]:




