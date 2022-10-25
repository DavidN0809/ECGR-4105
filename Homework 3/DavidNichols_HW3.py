#!/usr/bin/env python
# coding: utf-8

# In[20]:


#############################################################################
#David Nichols
#4105 HW 3
#https://github.com/DavidN0809/ECGR-4105/tree/main/Homework%203
#############################################################################


# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings("ignore")


# In[22]:


def metrics_print(y_pred, y_test):
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))

#Prints the matrix
def matrix_print(cnf_matrix):
    
    class_names = [0,1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1) 
    plt.ylabel('Actual label') 
    plt.xlabel('Predicted label')   
    
def create_pca(scled_x, raw_y, columns, n):
    
    pca = PCA(n_components = n)
    principalComponents = pca.fit_transform(scled_x) 
    principalDf = pd.DataFrame(data = principalComponents, columns = columns) 
    
    finalDf = pd.concat([principalDf, raw_y], axis = 1)
    return finalDf
    
    
def logistic_regression(raw_x, raw_y):
    #Splits the data
    x_train, x_test, y_train, y_test = train_test_split(raw_x, breast.target, test_size = 0.20, random_state = 5)
    
    #Creates model for Logistic Regression in terms of the data
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    metrics_print(y_pred, y_test)
    
    #CONFUSION MATRIX
    matrix = confusion_matrix(y_test, y_pred)
    print("Matrix: \n\n", matrix)
    
    return classifier, matrix


def graph_pca(data):
    fig = plt.figure(figsize = (12,12)) 
    ax = fig.add_subplot(1,1,1)  
    ax.set_xlabel('Principal Component 1', fontsize = 15) 
    ax.set_ylabel('Principal Component 2', fontsize = 15) 
    ax.set_title('2 component PCA', fontsize = 20) 
    targets = ['Malignant','Benign']
    colors = ['r', 'g', 'b'] 
    for target, color in zip(targets,colors): 
        indicesToKeep = pca_y == target 
        ax.scatter(data.loc[indicesToKeep, 'Principal Component 1'], data.loc[indicesToKeep, 'Principal Component 2'], c = color, s = 50) 
    ax.legend(targets) 
    ax.grid()


# In[23]:


breast = load_breast_cancer()
breast_data = breast.data
breast_data.shape

breast_input = pd.DataFrame(breast_data)
breast_input.head()

breast_labels = breast.target 
breast_labels.shape

breast_labels = np.reshape(breast_labels,(breast_labels.size,1))
final_breast_data = np.concatenate([breast_data,breast_labels],axis=1)
final_breast_data.shape

breast_dataset = pd.DataFrame(final_breast_data)
features_x = breast.feature_names
features_labels = np.append(features_x,'label')
breast_dataset.columns = features_labels
    
breast_dataset['label'].replace(0, 'Benign',inplace=True) 
breast_dataset['label'].replace(1, 'Malignant',inplace=True) 
breast_dataset.head()


# In[24]:


dataset=breast
model=GaussianNB();
model.fit(dataset.data, dataset.target)
print(model)
model.fit(dataset.data, dataset.target)
print(model)
expected = dataset.target
predicted = model.predict(dataset.data)
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))


# In[27]:


#Explain and elaborate on your results.
#Can you compare your results against the logistic regression classifier you did in previous homework.
#This is better due to the higher or same value or accuracy
#problem 2


# In[25]:


raw_x = breast_dataset[features_x]
raw_y = breast_dataset['label']


# In[26]:


sc_x = StandardScaler()
scled_x = sc_x.fit_transform(raw_x)

nonPCA_classifier, nonPCA_matrix = logistic_regression(scled_x, raw_y)
x_train, x_test, y_train, y_test = train_test_split(raw_x, breast.target, test_size = 0.20, random_state = 12)
    
classifier = LogisticRegression()
classifier.fit(x_train, y_train)    

y_pred = classifier.predict(x_test)

matrix_print(nonPCA_matrix)


# In[28]:


columns = ['Principal Component 1', 'Principal Component 2']
pca_data = create_pca(scled_x, raw_y, columns, 3)
pca_data


# In[29]:


pca_x = pca_data[columns]
pca_y = pca_data['label']

graph_pca(pca_data)


# In[30]:


pca_classifier, pca_matrix = logistic_regression(pca_x, pca_y)


# In[31]:


columns = ['1', '2', '3']
pca_data = create_pca(scled_x, raw_y, columns, 3)

pca_x = pca_data[columns]
pca_y = pca_data['label']

pca_classifier, pca_matrix = logistic_regression(pca_x, pca_y)


# In[32]:


columns = ['1', '2', '3', '4', '5','6']
pca_data = create_pca(scled_x, raw_y, columns, 6)

pca_x = pca_data[columns]
pca_y = pca_data['label']

pca_classifier, pca_matrix = logistic_regression(pca_x, pca_y)


# In[33]:


#Explain and elaborate on your results
#As you can see the higher number of K is better due to the higher values
#Problem 3


# In[34]:


NB_classifier = GaussianNB()
NB_classifier.fit(x_train, y_train)


# In[35]:


NB_y_pred = NB_classifier.predict(x_test)
NB_matrix = confusion_matrix (y_test, NB_y_pred)
print("NB_Matrix: \n\n", NB_matrix)


# In[36]:


metrics_print(NB_y_pred, y_test)


# In[37]:


matrix_print(NB_matrix)


# In[38]:


#Compare your results against problem 2
#As you can see the NB is worse due to the lower values 

