#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# In[7]:


df = pd.read_csv('D3.csv')
df.head() # To get first n rows from the dataset default value of n is 5 
M=len(df) 
M 


# In[8]:


X = df.values[:, 0]  # get input values from first column 
y = df.values[:, 1]  # get output values from second column 
m = len(y) # Number of training examples 
print('X = ', X[: 5]) # Show only first 5 records 
print('y = ', y[: 5]) 
print('m = ', m) 


# In[9]:


X = df.values[:, 0]  # get input values from first column 
y = df.values[:, 1]  # get output values from second column 
m = len(y) # Number of training examples 
print('X = ', X[: 97]) # Show only first 5 records 
print('y = ', y[: 97]) 
print('m = ', m) 


# In[10]:


plt.scatter(X,y, color='red',marker= '+') 
plt.grid() 
plt.rcParams["figure.figsize"] = (10,6) 
plt.xlabel('Population of City in 10,000s') 
plt.ylabel('Profit in $10,000s') 
plt.title('Scatter plot of training data') 


# In[11]:


#Lets create a matrix with single column of ones 
X_0 = np.ones((m, 1)) 
X_0[:5] 


# In[12]:


# Using reshape function convert X 1D array to 2D array of dimension 97x1 
X_1 = X.reshape(m, 1) 
X_1[:10] 


# In[13]:


# Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column 
# This will be our final X matrix (feature matrix) 
X = np.hstack((X_0, X_1)) 
X[:5] 


# In[14]:


theta = np.zeros(2) 
theta 


# In[15]:


def compute_cost(X, y, theta): 
  """ 
  Compute cost for linear regression. 
 
  Input Parameters 
  ---------------- 
  X : 2D array where each row represent the training example and each column represent 
      m= number of training examples 
      n= number of features (including X_0 column of ones) 
  y : 1D array of labels/target value for each traing example. dimension(1 x m) 
 
  theta : 1D array of fitting parameters or weights. Dimension (1 x n) 
 
  Output Parameters 
  ----------------- 
  J : Scalar value. 
  """ 
  predictions = X.dot(theta) 
  errors = np.subtract(predictions, y) 
  sqrErrors = np.square(errors) 
  J = 1 / (2 * m) * np.sum(sqrErrors) 
 
  return J 


# In[16]:


# Lets compute the cost for theta values 
cost = compute_cost(X, y, theta) 
print('The cost for given values of theta_0 and theta_1 =', cost) 


# In[17]:


def gradient_descent(X, y, theta, alpha, iterations): 
  """ 
  Compute cost for linear regression. 
 
  Input Parameters 
  ---------------- 
  X : 2D array where each row represent the training example and each column represent 
      m= number of training examples 
      n= number of features (including X_0 column of ones) 
  y : 1D array of labels/target value for each traing example. dimension(m x 1) 
  theta : 1D array of fitting parameters or weights. Dimension (1 x n) 
  alpha : Learning rate. Scalar value 
  iterations: No of iterations. Scalar value.  
 
  Output Parameters 
  ----------------- 
  theta : Final Value. 1D array of fitting parameters or weights. Dimension (1 x n) 
  cost_history: Conatins value of cost for each iteration. 1D array. Dimansion(m x 1) 
  """ 
  cost_history = np.zeros(iterations) 
 
  for i in range(iterations): 
    predictions = X.dot(theta) 
    errors = np.subtract(predictions, y) 
    sum_delta = (alpha / m) * X.transpose().dot(errors); 
    theta = theta - sum_delta; 
    cost_history[i] = compute_cost(X, y, theta)   
 
  return theta, cost_history 


# In[18]:


theta = [0., 0.] 
iterations = 1500; 
alpha = 0.01; 


# In[19]:


theta, cost_history = gradient_descent(X, y, theta, alpha, iterations) 
print('Final value of theta =', theta) 
print('cost_history =', cost_history) 


# In[20]:


# Since X is list of list (feature matrix) lets take values of column of index 1 only 
plt.scatter(X[:,1], y, color='red', marker= '+', label= 'Training Data') 
plt.plot(X[:,1],X.dot(theta), color='green', label='Linear Regression') 
 
plt.rcParams["figure.figsize"] = (10,6) 
plt.grid() 
plt.xlabel('Population of City in 10,000s') 
plt.ylabel('Profit in $10,000s') 
plt.title('Linear Regression Fit') 
plt.legend() 


# In[21]:


plt.plot(range(1, iterations + 1),cost_history, color='blue') 
plt.rcParams["figure.figsize"] = (10,6) 
plt.grid() 
plt.xlabel('Number of iterations') 
plt.ylabel('Cost (J)') 
plt.title('Convergence of gradient descent') 


# In[ ]:




