#!/usr/bin/env python
# coding: utf-8

# In[248]:


#Importing the required libraried to perform the given tasks

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as snshttp://localhost:8888/notebooks/PoojithaMandalapu_ML_Assignment4.ipynb#


# # 1. Principal Component Analysis
#     a. Apply PCA on CC dataset.
#     b. Apply k-means algorithm on the PCA result and report your observation if the silhouette score
#        has improved or not?
#     c. Perform Scaling+PCA+K-Means and report performance.

# In[249]:


#Loading the dataset

cc_dataset=pd.read_csv('datasets/CC.csv')
cc_dataset.head()


# In[250]:


#Applying the imputer to the dataset to fill the null values that will prevent the PCA

X = cc_dataset.iloc[:,1:]
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X)
X = imputer.transform(X)
X=pd.DataFrame(X)


# In[251]:


#a. Apply PCA on CC dataset

pca = PCA(2)
x_pca = pca.fit_transform(X)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2, X.iloc[:,-1]], axis=1)
finaldf.head()


# In[252]:


#Performing the elbow method to find the best number of suitable clusters for the given data to implement k-means
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(finaldf)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()


# In[253]:


# Apply k-means algorithm on the PCA result and report your observation if the silhouette score has improved or not?

nclusters = 4
km = KMeans(n_clusters=nclusters)
km.fit(finaldf)


# In[254]:


y_cluster_kmeans = km.predict(finaldf)
score = metrics.silhouette_score(finaldf, y_cluster_kmeans)
print('Silhoutte score for just PCA:',score)


# In[255]:


#Reload the dataset again 
X = cc_dataset.iloc[:,1:]

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X)
  
X = imputer.transform(X)

print(X)
X=pd.DataFrame(X)


# In[256]:


#Apply scaling on the dataset 

scaler = StandardScaler()
scaler.fit(X)
x_scaler = scaler.transform(X)

#Apply PCA with k value as 2 again

pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2,cc_dataset[['TENURE']]],axis=1)
print(finaldf)


# In[257]:


#Apply k-means on the scaled PCA output

nclusters = 4
km = KMeans(n_clusters=nclusters)
km.fit(finaldf)


# In[258]:


y_cluster_kmeans = km.predict(finaldf)
score = metrics.silhouette_score(finaldf, y_cluster_kmeans)
print('Silhoutte score for scaled=pca=keans:',score)


# #Observation: 
#     
#  The score is reduced after performing the PCa, so this data need not to be undergone with PCA.

# # 2. Use pd_speech_features.csv
#     a. Perform Scaling
#     b. Apply PCA (k=3)
#     c. Use SVM to report performance

# In[269]:


#Load the dataset

speech_df=pd.read_csv('datasets/pd_speech_features.csv')
speech_df.head()


# In[272]:


#Apply scaling on the dataset

x =speech_df.iloc[:,1:]
scaler = StandardScaler()
scaler.fit(x)
speech_x_scaler = scaler.transform(x)

#Apply PCA with value 3

pca = PCA(3)
speech_x_pca = pca.fit_transform(speech_x_scaler)
speech_df2 = pd.DataFrame(data=speech_x_pca)
speech_finaldf = pd.concat([speech_df2,speech_df[['class']]],axis=1)
print(speech_finaldf)


# In[273]:


#Apply SVM classifier

clf = SVC(kernel='linear') 
x =speech_finaldf.iloc[:,:-1]
y =speech_finaldf.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
accuracy_score(y_test, y_pred)
print("SVM accuracy =", accuracy_score(y_test, y_pred))


# In[274]:


#Classification report for the above classifier

print(classification_report(y_test, y_pred))


# # 3. Apply Linear Discriminant Analysis (LDA) on Iris.csv dataset to reduce dimensionality of data to k=2

# In[276]:


#Load the IRIS dataset

iris_df = pd.read_csv("datasets/iris.csv")
iris_df.head()


# In[281]:


#apply the standard scaling

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(iris_df.iloc[:,:-1].values)

#Label encoding the species column
class_le = LabelEncoder()
y = class_le.fit_transform(iris_df['Species'].values)

#Applying LDA on the Datset

lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_std,y)

data=pd.DataFrame(X_train_lda)
data['class']=y
data.columns=["LD1","LD2","class"]
data.head()


# In[284]:


markers = ['s', 'x', 'o']
colors = ['y', 'b', 'g']
sns.lmplot(x="LD1", y="LD2", data=data, hue='class', markers=markers, fit_reg=False, legend=False)
plt.legend()
plt.show()


# # 4. Briefly identify the difference between PCA and LDA
# 
# Answer: PCA performs better in case where number of samples per class is less. Whereas LDA works better with large dataset having      multiple classes; class separability is an important factor while reducing dimensionality. PCA finds directions of maximum      variance regardless of class labels while LDA finds directions of maximum class separability.
