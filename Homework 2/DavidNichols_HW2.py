#!/usr/bin/env python
# coding: utf-8

# In[433]:


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
from sklearn import metrics 
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")


# In[434]:


iris = datasets.load_iris()
X=iris.data
y=iris.target


# In[435]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)


# In[436]:


sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# In[437]:


C = [10, 1, 0.1, 0.001]
for c in C:
    clf = LogisticRegression(penalty='l1', C=c, solver='liblinear')
    clf.fit(X_train, y_train)
    print('C:',c)
    print('Training accuracy', clf.score(X_train_std, y_train))
    print('Test accuracy:', clf.score(X_test_std,y_test))
    print('')


# In[438]:


kfold = KFold(n_splits=5, random_state=0, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[439]:


test_size = 0.80
seed = 0
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)


# In[440]:


test_size = 0.80
seed = 0;
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)


# In[441]:


class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(matrix), annot = True, cmap = "YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual Label')
plt.xlabel('Predicted label')


# In[442]:


#GIVEN CODE
#Problem 1


# In[443]:


raw_data = pd.read_csv('diabetes.csv')
raw_data


# In[444]:


def matrix_print(matrix):
    class_names=[0,1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks,class_names)
    sns.heatmap(pd.DataFrame(matrix), annot = True, cmap = "YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted label')


# In[445]:


def metrics_print(y_pred):
    print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
    print("Precision:", metrics.precision_score(Y_test, y_pred))
    print("Recall:", metrics.recall_score(Y_test, y_pred))


# In[446]:


varlist = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
raw_x = raw_data[varlist]
raw_y = raw_data['Outcome']


# In[447]:


X_train, X_test, Y_train, Y_test = train_test_split(raw_x, raw_y, test_size = 0.20, random_state = 5)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#scaling and transforming before classifying


# In[448]:


classifier = LogisticRegression(random_state=0) 
classifier.fit(X_train, Y_train)


# In[449]:


Y_pred = classifier.predict(X_test) 


# In[450]:


Y_pred[0:9] 


# In[451]:


cnf_matrix = confusion_matrix(Y_test, Y_pred) 
cnf_matrix 


# In[452]:


metrics_print(Y_pred)


# In[453]:


matrix_print(cnf_matrix)


# In[454]:


#problem 2


# In[455]:


kfold = KFold(n_splits=5, random_state=0, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(classifier, X_train, Y_train, cv=kfold)
print("Accuracy of 5 kfold: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[456]:


kfold = KFold(n_splits=10, random_state=0, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(classifier, X_train, Y_train, cv=kfold)
print("Accuracy of 10 kfold: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[457]:


print("Accuracy of problem 1")
metrics_print(Y_pred)
#As you can see problems 1 accuracy is higher


# In[458]:


NB_classifier = GaussianNB()
NB_classifier.fit(X_train, Y_train)
NB_Y_predicition = NB_classifier.predict(X_test)
NB_Matrix = confusion_matrix(Y_test, NB_Y_predicition)
matrix_print(NB_Matrix)


# In[459]:


#problem 3


# In[460]:


from sklearn.datasets import load_breast_cancer 
breast = load_breast_cancer()
breast_data = breast.data
breast_data.shape


# In[461]:


breat_input=pd.DataFrame(breast_data)
breat_input.head()


# In[462]:


breast_labels = breast.target
breast_labels.shape


# In[463]:


labels = np.reshape(breast_labels, (569,1))
final_breast_data = np.concatenate([breast_data,labels], axis=1)
final_breast_data.shape


# In[464]:


breast_dataset= pd.DataFrame(final_breast_data)
features = breast.feature_names
features_labels = np.append(features, 'label')
breast_dataset.columns = features_labels
breast_dataset


# In[465]:


breast_dataset['label'].replace(0,'Benign',inplace=True)
breast_dataset['label'].replace(1,'Malignant',inplace=True)


# In[466]:


X_train, X_test, Y_train, Y_test = train_test_split(raw_x, raw_y, test_size = 0.20, random_state = 5)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#scaling and transforming before classifying

classifier = LogisticRegression(random_state=0) 
classifier.fit(X_train, Y_train)


# In[467]:


Y_pred = classifier.predict(X_test) 
Y_pred[0:9] 


# In[468]:


cnf_matrix = confusion_matrix(Y_test, Y_pred) 
cnf_matrix 


# In[469]:


metrics_print(Y_pred)


# In[470]:


matrix_print(cnf_matrix)


# In[471]:


#Problem 4


# In[472]:


kfold = KFold(n_splits=5, random_state=0, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(classifier, X_train, Y_train, cv=kfold)
print("Accuracy of 5 kfold: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[473]:


kfold = KFold(n_splits=10, random_state=0, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(classifier, X_train, Y_train, cv=kfold)
print("Accuracy of 10 kfold: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[ ]:




