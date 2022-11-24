#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#############################################################################
#David Nichols
#4105 HW 3
#https://github.com/DavidN0809/ECGR-4105/tree/main/Homework%205
#############################################################################


# In[244]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import tensorflow as tf
from matplotlib import pyplot as plt
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")


# In[207]:


def loss_fn(t_p,t_c):
    squared_diffs = (t_p-t_c)**2
    return squared_diffs.mean()


# In[208]:


def dloss_fn(t_p, t_c_):
    dsq_diffs = 2 * (t_p - t_c) /t_p.size(0)
    return dsq_diffs


# In[209]:


def dmodel_dw1(t_u, w1, w2, b):
    return t_u

def dmodel_dw2(t_u, w1, w2, b):
    return t_u


# In[210]:


def dmodel_db(t_u, w1, w2, b):
    return 1.0


# In[211]:


def grad_fn(t_u, t_c, t_p, w1, w2, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw1 = dloss_dtp * dmodel_dw1(t_u, w1, w2, b)
    dloss_dw2 = dloss_dtp * dmodel_dw2(t_u, w1, w2, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w1, w2, b)
    return torch.stack([dloss_dw1.sum(), dloss_dw2.sum(), dloss_db.sum()])


# In[212]:


#problem 1


# In[213]:


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


# In[214]:


def new_model(t_u, w1, w2, b):
    return w2 * t_u **2 + w1 *t_u + b


# In[215]:


w1 = torch.ones(()) #initial W is 1
w2 = torch.ones(()) #initial W is 1
b = torch.zeros(()) #initial b is 0
t_p = new_model(t_u, w1, w2, b)
t_p


# In[216]:


loss = loss_fn(t_p, t_c)
loss


# In[217]:


t_un = 0.1 *t_u


# In[218]:


def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs+1):
        t_p = new_model(t_u, *params)
        loss=loss_fn(t_p,t_c)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params


# In[219]:


params = torch.tensor([2.0,1.0,0.0])
params.requires_grad=True
learning_rate = 0.00001
optimizer = optim.SGD([params],lr=learning_rate)

training_loop(
n_epochs = 5000,
optimizer = optimizer,
params = params,
t_u = t_un,
t_c = t_c)
    
t_p = new_model(t_un, *params)
fig = plt.figure(dpi=600)
plt.xlabel=("Temperature (Fahrenheigh)")
plt.ylabel=("Temperature (Celsuis)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(),'o')


# 1.c 
# None linear line fits the data better, hence letting it have a more accurate loss and prediction. 
# 

# Problem 2

# In[220]:


device = torch.device("cuda:0")
housing = pd.DataFrame(pd.read_csv('Housing.csv'))
housing.head()


# In[221]:


varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
def binary_map(x):
    return x.map({'yes' : 1, 'no' : 0, 'furnished' : 1, 'semi-furnished' : 0.5, 'unfurnished' : 0 })
housing[varlist] = housing[varlist].apply(binary_map)
housing.head()


# In[222]:


features = ['price','area', 'bedrooms', 'bathrooms', 'stories', 'parking']
dataset = housing[features]
dataset = StandardScaler().fit_transform(dataset)
dataset


# In[223]:


raw_y = dataset[:, 0]
raw_x=dataset[:,1:6]
x=torch.from_numpy(raw_x)
y=torch.from_numpy(raw_y)


# In[224]:


n_samples = x.shape[0]
n_val = int(0.2 * n_samples)
shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]
#train_indices, val_indices


# In[225]:


def housing_model(X,W1,W2,W3,W4,W5,B):
    U=W5*X[:,4] + W4*X[:,3] + W3*X[:,2] + W2*X[:,1] + W1*X[:,0] + B
    return U

def loss_fn(Yp, Y):
    squared_diffs = (Yp - Y)**2
    return squared_diffs.mean()


# In[226]:


train_t_u = x[train_indices]
train_t_c = y[train_indices]
val_t_u = x[train_indices]
val_t_c = y[train_indices]
train_t_un = 0.1 * train_t_u
val_t_un = 0.1*val_t_u


# In[227]:


def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u, 
                  train_t_c, val_t_c):
    for epoch in range(1,n_epochs+1):
        train_t_p = housing_model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)
        
        val_t_p = housing_model(val_t_u, *params)
        val_loss = loss_fn(val_t_p, val_t_c)
        
        with torch.no_grad():
            val_t_p = housing_model(val_t_u, *params)
            val_loss = loss_fn(val_t_p, val_t_c)
            assert val_loss.requires_grad == False
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if epoch <= 3 or epoch % 500 == 0:
            print (f"epoch {epoch}, Training loss {train_loss.item():.4f}," f" Validation loss {val_loss.item():.4f}")
                   
    return params


# In[228]:


def training_SGD(lr):
    params=torch.tensor([1.0,1.0,1.0,1.0,1.0,0.0],requires_grad=True)
    learning_rate=lr
    optimizer=optim.SGD([params],lr=learning_rate)

    training_loop(
        n_epochs = 5000,
        optimizer = optimizer,
        params = params,
        train_t_u = train_t_un,
        val_t_u = val_t_un,
        train_t_c = train_t_c,
        val_t_c = val_t_c)


# In[229]:


training_SGD(0.0001)


# In[230]:


training_SGD(0.001)


# In[231]:


training_SGD(0.01)


# In[232]:


training_SGD(0.1)


# In[233]:


def training_Adam(lr):
    params=torch.tensor([1.0,1.0,1.0,1.0,1.0,0.0],requires_grad=True)
    learning_rate=lr
    optimizer=optim.Adam([params],lr=learning_rate)

    training_loop(
        n_epochs = 5000,
        optimizer = optimizer,
        params = params,
        train_t_u = train_t_un,
        val_t_u = val_t_un,
        train_t_c = train_t_c,
        val_t_c = val_t_c)


# In[234]:


training_Adam(0.1)


# The best linear model is adam, due to its lower loss and it reaches this loss faster.

# Problem 3

# In[235]:


t_u_train = train_t_u
t_c_train = train_t_c

t_u_val = val_t_u
t_c_val = val_t_c

t_un_train = train_t_un
t_un_val = val_t_un


# In[236]:


linear_model = nn.Linear(1 , 1)

optimizer = optim.SGD(
    linear_model.parameters(), # <2>
    lr=1e-2)

seq_model = nn.Sequential(
            nn.Linear(5, 8), # <1>
            nn.Tanh(),
            nn.Linear(8, 1)) # <2>

seq_model = seq_model.double()


# In[237]:


[param.shape for param in seq_model.parameters()]

for name, param in seq_model.named_parameters():
    print(name, param.shape)


# In[238]:


def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val,
                  t_c_train, t_c_val):
    for epoch in range(1, n_epochs + 1):
        t_p_train = model(t_u_train) # <1>
        loss_train = loss_fn(t_p_train, t_c_train)

        t_p_val = model(t_u_val) # <1>
        loss_val = loss_fn(t_p_val, t_c_val)
        
        optimizer.zero_grad()
        loss_train.backward() # <2>
        optimizer.step()

        if epoch == 1 or epoch % 50 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                  f" Validation loss {loss_val.item():.4f}")
            
def loss_fn(Yp, Y):
    squared_diffs = (Yp - Y)**2
    return squared_diffs.mean()


# In[245]:


optimizer = optim.SGD(seq_model.parameters(), lr=1e-3) # <1>

training_loop(
    n_epochs = 200, 
    optimizer = optimizer,
    model = seq_model,
    loss_fn = nn.MSELoss(),
    t_u_train = t_un_train,
    t_u_val = t_un_val, 
    t_c_train = t_c_train,
    t_c_val = t_c_val)


# As epochs increase te loss gets less. This is due to a longer time

# Problem 3b

# In[240]:


t_u_train = train_t_u
t_c_train = train_t_c

t_u_val = val_t_u
t_c_val = val_t_c

t_un_train = train_t_un
t_un_val = val_t_un


# In[241]:


linear_model = nn.Linear(1 , 1)

optimizer = optim.SGD(
    linear_model.parameters(), # <2>
    lr=1e-2)

seq_model = nn.Sequential(
            nn.Linear(5, 8), # <1>
            nn.Tanh(),
            nn.Linear(8, 4), # <2>
            nn.Tanh(),
            nn.Linear(4, 2), # <3>
            nn.Tanh(),
            nn.Linear(2, 1))

seq_model = seq_model.double()


# In[242]:


[param.shape for param in seq_model.parameters()]

for name, param in seq_model.named_parameters():
    print(name, param.shape)


# In[243]:


optimizer = optim.SGD(seq_model.parameters(), lr=1e-3) # <1>

training_loop(
    n_epochs = 200, 
    optimizer = optimizer,
    model = seq_model,
    loss_fn = nn.MSELoss(),
    t_u_train = t_un_train,
    t_u_val = t_un_val, 
    t_c_train = t_c_train,
    t_c_val = t_c_val)


# Adding more layers to the network allowed for a better loss values, this is due to how to network processes the inputs.

# In[ ]:




