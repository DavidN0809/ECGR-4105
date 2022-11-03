#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#############################################################################
#David Nichols
#4105 HW 4
#https://github.com/DavidN0809/ECGR-4105/tree/main/Homework%203
#############################################################################


# In[1]:


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
# Importing required libraries
from seaborn import load_dataset, pairplot
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")


# In[2]:


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

def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    ax.contour(X, Y, P, colors='k', 
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--','-','--'])
    if plot_support:
        ax.scatter(model.support_[:, 0], model.support_[:, 1],
                  s=300, linewidth = 1, facecolors='none');
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    plt.scatter(X[:,0],X[:,1], c=y, s=50, cmap = 'autumn')


# In[3]:


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
    
#breast_dataset['label'].replace(0, 'Benign',inplace=True) 
#breast_dataset['label'].replace(1, 'Malignant',inplace=True) 
breast_dataset.head()


# In[4]:


df = breast_dataset
df.head()


# In[5]:


raw_x = breast_dataset[features_x]
raw_y = breast_dataset['label']


# In[6]:


#part 1


# In[7]:


sc_x = StandardScaler()
scled_x = sc_x.fit_transform(raw_x)

accuracyl = list()
recalll = list()
precisionl = list()
def pcaselect(kernal,c=10):
    for n in range(1,31):
        pca = PCA(n)
        principalComponents = pca.fit_transform(scled_x)
        principalDf = pd.DataFrame(data = principalComponents)
        finalDf = pd.concat([principalDf, df[['label']]], axis = 1)
        x_train, x_test, y_train, y_test = train_test_split(principalDf, breast.target, test_size = 0.20, random_state = 12)
        classifier = SVC(kernel=kernal, C=c, random_state = 12)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracyl.append(metrics.accuracy_score(y_test, y_pred))
        precisionl.append(metrics.precision_score(y_test, y_pred))
        recalll.append( metrics.recall_score(y_test, y_pred))


# In[8]:


#part 2 graph
#part 3
#The kernal rbf was selected due to it giving me the highest graphs


# In[9]:


pcaselect('rbf',c=10)

plt.plot(accuracyl)
plt.plot(precisionl)
plt.plot(recalll)
    
plt.grid()
plt.xlabel('data')
plt.ylabel('target')
plt.title('SVM')
plt.show()


# In[10]:


n=8;
pca = PCA(n)
principalComponents = pca.fit_transform(scled_x)
principalDf = pd.DataFrame(data = principalComponents)
finalDf = pd.concat([principalDf, df[['label']]], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(principalDf, breast.target, test_size = 0.20, random_state = 12)
classifier = SVC(kernel='rbf', C=10, random_state = 12)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
accuracy=(metrics.accuracy_score(y_test, y_pred))
precision=(metrics.precision_score(y_test, y_pred))
recall=( metrics.recall_score(y_test, y_pred))


# In[11]:


print(accuracy)
print(precision)
print(recall)


# In[12]:


#problem 2


# In[13]:


housing = pd.DataFrame(pd.read_csv('Housing.csv'))

varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
def binary_map(x):
    return x.map({'yes' : 1, 'no' : 0, 'furnished' : 1, 'semi-furnished' : 0.5, 'unfurnished' : 0 })
housing[varlist] = housing[varlist].apply(binary_map)

housing.head()


# In[14]:


del housing['furnishingstatus']


# In[15]:


housing.head()


# In[16]:


features = ['area','bedrooms','bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating','airconditioning','parking','prefarea']
housing = StandardScaler().fit_transform(housing)
housing=pd.DataFrame(housing)
y = housing.pop(0)
x = housing


# In[25]:



test_r2_list = list()
training_r2_list = list()
test_MAE_list = list()
training_MAE_list = list()
training_MSE_list = list()
test_MSE_list = list()
Max_Error_list = list()


def svr_select(kernel='rbf',c=10):
    for n in range(1,12):
        pca = PCA(n)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents)
        #finalDf = pd.concat([principalDf, df[['label']]], axis = 1)
        x_train, x_test, y_train, y_test = train_test_split(principalDf, y, test_size = 0.20,)
        classifier = SVR(kernel='rbf', C=10)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
    
        test_r2_list.append(r2_score(y_test, y_pred))
        training_r2_list.append(classifier.score(x_train, y_train))

        test_MAE_list.append(mean_absolute_error(y_test, y_pred))
        training_MAE_list.append(mean_absolute_error(y_train, classifier.predict(x_train)))

        training_MSE_list.append(mean_squared_error(y_test, y_pred))
        test_MSE_list.append(mean_squared_error(y_train, classifier.predict(x_train)))
            
        Max_Error_list.append(max_error(y_test, y_pred))

                
def graph_svr(kernel='kernel'):
    
    plt.plot(test_r2_list,label = 'test_r2')
    plt.plot(training_r2_list,label = 'training_r2')

    plt.plot(test_MAE_list,label = 'test_MAE')
    plt.plot(training_MAE_list,label = 'training_MAE')

    plt.plot(training_MSE_list,label = 'training_MSE')
    plt.plot(test_MSE_list,label = 'test_MSE')


    plt.grid()
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title(kernel)
    plt.legend()
    plt.show()


# In[27]:


svr_select(kernel='rbf')
graph_svr(kernel='rbf')


svr_select(kernel='linear')
graph_svr(kernel='linear')


svr_select(kernel='poly')
graph_svr(kernel='poly')
    


# In[40]:


#poly is best due to higher peaks at 8


# In[41]:


n=8
pca = PCA(n)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)
#finalDf = pd.concat([principalDf, df[['label']]], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(principalDf, y, test_size = 0.20,)
classifier = SVR(kernel='poly', C=10)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)    
test_r2_list.append(r2_score(y_test, y_pred))

training_r2 = (classifier.score(x_train, y_train))
test_MAE_list = (mean_absolute_error(y_test, y_pred))
training_MAE_list = (mean_absolute_error(y_train, classifier.predict(x_train)))
training_MSE_list = (mean_squared_error(y_test, y_pred))
test_MSE_list = (mean_squared_error(y_train, classifier.predict(x_train)))         
Max_Error_list = (max_error(y_test, y_pred))

print('training r2', training_r2)
print('test MAE', test_MAE_list)
print('training MAE', training_MAE_list)
print('training MSE', training_MSE_list)
print('test_MSE', test_MSE_list)
print('Max Error', Max_Error_list)


# In[ ]:




