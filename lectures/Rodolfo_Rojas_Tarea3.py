#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import torch
import torch.optim as optim


# In[2]:


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] # Temperatura en grados celsios
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] # Unidades desconocidas
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


# In[4]:


def model(t_u, w2, w1, b):
    return w2 * t_u ** 2 + w1 * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[9]:


def training_loop(model, n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss {loss}")
            
    return params


# In[10]:


params = torch.tensor([1.0, 1.0, 0.0], requires_grad=True)
learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate) # Nuevo optimizador


# In[11]:


training_loop(model,
              n_epochs=2000,
              optimizer=optimizer,
              params = params,
              t_u = t_u, # Regresamos a usar el t_u original como input
              t_c = t_c)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb')

