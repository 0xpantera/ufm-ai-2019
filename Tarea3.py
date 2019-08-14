#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import torch
import torch.optim as optim


# In[2]:


def model(t_u, w1, w2, b):
    return w2 * t_u ** 2 + w1 * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[3]:


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



t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] # Temperatura en grados celsios
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] # Unidades desconocidas
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


params = torch.tensor([1.0, 1.0, 0.0], requires_grad=True)
learning_rate = 1e-4
optimizer = optim.SGD([params], lr=learning_rate)

t_un = 0.1 * t_u

training_loop(model,
              n_epochs=5000,
              optimizer=optimizer,
             params = params, # Es importante que ambos `params` sean el mismo objeto
             t_u = t_un,
             t_c = t_c)


# **Que partes del training loop necesitaron cambiar para acomodar el nuevo modelo?**

# Del *Training Loop* ninguno ya que acepta el argunmento *params* como un array donde se le pueden agregar un mayor numero de tensores.   
# Se realizaron cambios en la funcion *model* para que aceptara mas parametros

# **Que partes se mantuvieron iguales?**

# El training Loop se mantuvo igual 

# **El loss resultante es mas alto o bajo despues de entrenamiento?**

# El nuevo *Loss* resultante tiene valor de **3.86174**, menor al *Loss* del modelo anterior con un valor de **2.92764**

# **El resultado es mejor o peor?**

# Peor ya que se tiene un valor mayor, y el objetivo es minimizar el resultado de la funcion *Loss* para tener un mejor *accuracy* de los resultados

# In[ ]:




