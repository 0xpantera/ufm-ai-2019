#!/usr/bin/env python
# coding: utf-8

# ## Ricardo Pineda | 20160164

# ### Instrucciones

# ##### Redefinan el model a w2 * t_u ** 2 + w1 * t_u + b
# 
# Que partes del training loop necesitaron cambiar para acomodar el nuevo modelo?
# 
# Que partes se mantuvieron iguales?
# 
# El loss resultante es mas alto o bajo despues de entrenamiento?
# 
# El resultado es mejor o peor?

# ### Codigo

# In[5]:


import numpy as np
import torch
import torch.optim as optim


# In[11]:


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] # Temperatura en grados celsios
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] # Unidades desconocidas
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


# In[12]:


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


# In[13]:


def model(t_u, w1, w2, b):
    return w2 * t_u ** 2 + w1 * t_u + b


# In[14]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[16]:


params = torch.tensor([1.0 , 1.0, 0.0], requires_grad=True)
learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate)

training_loop(model, n_epochs=2000, optimizer=optimizer, params = params, t_u = t_u, t_c = t_c)


# ## Respuestas

# __Que partes del training loop necesitaron cambiar para acomodar el nuevo modelo?__
# 
# Lo unico que se cambia es la parte de 'model' para que acepte un parametro mas

# __Que partes se mantuvieron iguales?__
# 
# - Funcion de perdida
# - Optimizador
# - Learning Rate
# - Entrenamiento

# __El loss resultante es mas alto o bajo despues de entrenamiento?__
# 
# El loss bajo despues de entrenar la funcion

# __El resultado es mejor o peor?__
# 
# El error es mas alto que con la funcion original, por lo que podemos deducir que el modelo es peor

# In[ ]:




