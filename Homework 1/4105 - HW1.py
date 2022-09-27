#!/usr/bin/env python
# coding: utf-8

# In[35]:


#######################################################################################
#David Nichols
#4105 HW 1
#https://github.com/DavidN0809/ECGR-4105/blob/main/Homework%201/4105%20-%20HW1.py
#######################################################################################


# In[36]:


#######################################################################################
#Given Code
#######################################################################################


# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[38]:


housing = pd.DataFrame(pd.read_csv("Housing.csv"))
housing.head()


# In[39]:


m=len(housing)
m


# In[40]:


varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
def binary_map(x):
    return x.map({'yes' : 1, 'no' : 0, 'furnished' : 1, 'semi-furnished' : 0.5, 'unfurnished' : 0 })

housing[varlist] = housing[varlist].apply(binary_map)


# In[41]:


from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train,df_test = train_test_split(housing, train_size=0.8, test_size=0.2)


# In[42]:


num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
df_Newtrain = df_train[num_vars]
df_Newtest = df_test[num_vars]


# In[43]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
df_Newtrain[num_vars] = scaler.fit_transform(df_Newtrain[num_vars])


# In[44]:


y_Normtrain = df_Newtrain.pop('price')
X_Normtrain = df_Newtrain.copy() 


# In[45]:


Y= y_Normtrain.values


# In[46]:


X0 = df_Newtrain.values[:,0]


# In[47]:


#######################################################################################
#Problem 1
#######################################################################################


# In[48]:


def compute_cost(X, Y, theta, penalty = 0):
    predictions = X.dot(theta)
    errors = predictions - Y
    sqrErrors = np.square(errors)
    
    if (penalty == 0):
        J = 1 / (2 * m) * np.sum(sqrErrors) 
    else:
        J = 1/(2*m) * (np.sum(sqrErrors) + penalty * (np.sum(theta) - theta[0]))
    return J


# In[49]:


def DisplayData(X, color):
    plt.scatter(X,Y,color = color)
    plt.grid()
    plt.title('Scatter plot of Price of Home')
    plt.rcParams["figure.figsize"] = (10,6)
    plt.xlabel('area')
    plt.ylabel('price')
    
def gradient(cost_history, color):
    plt.plot(range(1,iterations+1),cost_history,color = color)
    plt.rcParams["figure.figsize"] = (10,6)
    plt.grid()
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.title('Convergence of gradient descent')


# In[50]:


#Redefined for problem 3
def gradient_descent(X, Y, X2, Y2, theta, alpha, iterations, penalty = 0):
    cost_history = np.zeros(iterations)
    cost_history2 = np.zeros(iterations)
    
    for i in range(iterations):
        prediction = X.dot(theta)
        errors = np.subtract(prediction,Y)
        sum_delta = (alpha/m)*X.transpose().dot(errors);
        
        if (penalty == 0):
            theta = theta - sum_delta;
        else:
            theta = theta * (1 - alpha * (penalty / m)) - sum_delta;
        
        cost_history[i] = compute_cost(X,Y,theta, penalty)
        cost_history2[i] = compute_cost(X2, Y2, theta)
        
    return theta, cost_history, cost_history2


# In[51]:


#Redefined for problem 3
def multiGraph(size, X, Y, X2, Y2, alpha = 0.01, penalty = 0):
    X0 = np.ones((len(X), 1))
    X = np.hstack((X0, X))
    
    X0 = np.ones((len(X2), 1))
    X2 = np.hstack((X0, X2))
    
    theta = np.zeros(size)
    iterations = 1500;
    
    cost = compute_cost(X, Y, theta)
    theta, cost_history, cost_history2 = gradient_descent(X, Y, X2, Y2, theta, alpha, iterations, penalty)
    
    return theta, cost_history, cost_history2


# In[52]:


raw_data = pd.DataFrame(pd.read_csv("Housing.csv"))


# In[53]:


m = len(raw_data)
raw_data[varlist] = raw_data[varlist].apply(binary_map)


# In[54]:


#Used to split the training and test
from sklearn.model_selection import train_test_split
np.random.seed(0)
#recalling to add random state
train_data, test_data = train_test_split(raw_data, train_size=0.8, test_size = 0.2, random_state = 42)


# In[55]:


def preprocessing(inputvars, data, Select = 'False'):
    data = data[inputvars]
    
    if Select == 'normalization':
        scaler = MinMaxScaler()
        data[inputvars] = scaler.fit_transform(data[inputvars])
        
    if Select == 'standardization':
        scaler = StandardScaler()
        data[inputvars] = scaler.fit_transform(data[inputvars])
        
    Y = data.pop('price')
    X = data
    
    return X, Y

def Everything(inputvars, Norm, alpha = 0.01, penalty = 0):
    X_train, Y_train = preprocessing(inputvars, train_data, Norm)
    X_test, Y_test = preprocessing(inputvars, test_data, Norm)

    theta_Train, cost_history_Train, cost_history_Test = multiGraph(len(inputvars), X_train, Y_train, X_test, Y_test, alpha, penalty)
    plt.rcParams["figure.figsize"] = (10,6)
    plt.grid()
    gradient(cost_history_Train, 'red')
    gradient(cost_history_Test, 'blue')
    
    return theta_Train


# In[56]:


input1 = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
input2 = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea']
iterations = 1500


# In[57]:


#raw_data.head()


# In[58]:


#1a
theta = Everything(input1, 'False', 0.0000000002)
plt.title('Problem 1a')
print('below is the best paramaters for 1a')
print('\n'.join('{}: {}'.format(*val) for val in enumerate(theta)))
#training is red and testing is blue
#The graph is broken unless a very low value is used.


# In[59]:


#1b
theta = Everything(input2, 'False', 0.0000000002)
plt.title('Problem 1b')
print('below is the best paramaters for 1b')
print('\n'.join('{}: {}'.format(*val) for val in enumerate(theta)))
#training is red and testing is blue
#The graph is broken unless a very low value is used.


# In[60]:


#######################################################################################
#Problem 2
#######################################################################################


# In[61]:


#2a
theta_Train = Everything(input1, 'normalization', 0.01)
plt.title('Problem 2a')
print('below is the best paramaters for 2a')
print('\n'.join('{}: {}'.format(*val) for val in enumerate(theta)))
print('\nWhich input scaling achieves the best training? Explain your results.')
print('Normalization achieves the best training due to its slope.')
print('\nCompare your training accuracy between both scaling approaches as well as the baseline training in problem 1 a.')
print('Normalization is better due to the higher alpha value, and normalization vs standardization, normalization is better due to its slope')


# In[62]:


#2b
theta_Train = Everything(input2, 'standardization')
plt.title('Problem 2b')
print('below is the best paramaters for 2b')
print('\n'.join('{}: {}'.format(*val) for val in enumerate(theta)))
print('\nWhich input scaling achieves the best training? Explain your results.')
print('Normalization achieves the best training due to its slope.')
print('\nCompare your training accuracy between both scaling approaches as well as the baseline training in problem 1 b.')
print('Normalization is better due to the higher alpha value, and normalization vs standardization, normalization is better due to its slope')


# In[63]:


#######################################################################################
#Problem 3
#######################################################################################


# In[64]:


#3a
theta = Everything(input1, 'normalization', 0.01, 3)
plt.title('Problem 3a')
print('\n'.join('{}: {}'.format(*val) for val in enumerate(theta)))
theta
print('\nExplain your results and compare them against problem 2 a')
print('The graphs are near identical for the test case. The training case without a penalty is better due to its lower cost.')


# In[65]:


#3b
theta = Everything(input2, 'standardization', 0.01, 3)
plt.title('Problem 3b')
print('\n'.join('{}: {}'.format(*val) for val in enumerate(theta)))
print('\nExplain your results and compare them against problem 2 b.')
print('The graphs are near identical for the test case. The training case with a penalty is better due to its lower cost.')


# In[ ]:




