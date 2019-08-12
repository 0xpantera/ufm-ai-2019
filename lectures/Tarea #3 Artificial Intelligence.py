#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.optim as optim


# In[12]:


#Definir el modelo original
def model(t_u, w, b):
    return w * t_u + b
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()
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
#Definir variables del modelo original
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] 
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] 
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
t_un = 0.1 * t_u
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)
#Ejecutar el modelo original
training_loop(model,
              n_epochs=5000,
              optimizer=optimizer,
              params = params,
              t_u = t_un,
              t_c = t_c)


# In[19]:


#Realizar los cambios a las variables pertinentes
#Redefinir el modelo para acoplar las nuevas variables
def model(t_u, w2, w1, b):
    return w2 * t_u ** 2 + w1 * t_u + b
#Redefinir los parametros para acoplar las nuevas variables 
params = torch.tensor([1.0, 1.0, 0.0], requires_grad=True)
#Redefinir el optimizador para acoplar el cambio en los parametros y reajustar el learning rate
learning_rate = 1e-4
optimizer = optim.SGD([params], lr=learning_rate)


# In[11]:


#Aparte de las tres partes que se cambiaron en la línea superior, el resto del training loop se mantiene igual 
#e.j. (loss_fn, optimizer.zero_grad, t_p, etc.)


# In[22]:


#Ejecutar el modelo con los cambios
training_loop(model,
              n_epochs=5000,
              optimizer=optimizer,
              params = params,
              t_u = t_un,
              t_c = t_c)


# In[ ]:


#Como se puede ver por los resultados, el modelo original lineal tiene un loss menor 
#que el loss del modelo polinomial de grado 2, por una diferencia de aproximadamente 0.49.

#Es gracias a este resultado que podemos concluir que el modelo original lineal tiene mejores resultados que el modelo
#nuevo.

