#!/usr/bin/env python
# coding: utf-8

# # IRIS FLOWER CLASSIFICATION

# # Dataset information

# The dataset contains 3 species of Iris flower: setosa, versicolor and virginia, which differs according to their measurements. Attribute Information:
# 
# 1.sepal length in cm
# 2.sepal width in cm
# 3.petal length in cm
# 4.petal width in cm
# 5.Species: setosa, versicolor, virginia

# # Importing Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


# # Importing datasets

# In[5]:


iris_data = pd.read_csv('IrisFlower.csv')
iris_data.head()


# # Preprocessing dataset

# In[6]:


type(iris_data)


# In[13]:


# deleting the column
iris_data=iris_data.drop(columns=['Id'])
iris_data.head()


# In[14]:


iris_data.iloc[50:100]


# # Data Analysis

# In[15]:


iris_data.describe()


# In[16]:


sns.pairplot(iris_data, hue="Species")
plt.show()


# In[17]:


# dropping class from column
X= iris_data.drop("Species", axis=1)
X
# so now x only contains features


# In[19]:


# now y represents the independent variables
y = iris_data["Species"]
y


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[21]:


X_train


# In[22]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)


# In[23]:


y_pred = knn.predict(X_test)


# In[24]:


print("Accuracy:", accuracy_score(y_test,y_pred))


# In[ ]:


print(classification_report(y_test, y_pred))


# In[25]:


X_test.head(2)


# In[29]:


new_data = pd.DataFrame({"SepalLengthCm":[5.1], "SepalWidthCm":[3.5], "PetalLengthCm":[1.4], "PetalWidthCm":[0.2]})


# In[30]:


prediction = knn.predict(new_data)


# In[31]:


prediction[0]


# In[32]:


new_data = pd.DataFrame({"SepalLengthCm":[5.7], "SepalWidthCm":[4.5], "PetalLengthCm":[2.4], "PetalWidthCm":[0.9]})
prediction = knn.predict(new_data)
prediction[0]


# In[ ]:


new_data = pd.DataFrame({"sepal_length":[6.7], "sepal_width":[6.5], "petal_length":[3.4], "petal_width":[2.9]})
prediction = knn.predict(new_data)
prediction[0]


# In[ ]:




