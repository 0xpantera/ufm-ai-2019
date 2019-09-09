#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
import torch.optim as optim


# In[5]:


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] # Temperatura en grados celsios
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] # Unidades desconocidas
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
t_un = t_u*0.1


# In[15]:


def model(t_un, w2, w1, b):
    return w2 * t_un ** 2 + w1 * t_un + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[19]:


def training_loop(model, n_epochs, optimizer, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_un, *params)
        loss = loss_fn(t_p, t_c)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss {loss}")
            
    return params
       
        
        


# In[20]:


params = torch.tensor([1.0, 1.0, 0.0], requires_grad=True)
w2, w1, b = params
learning_rate = 1e-4
optimizer = optim.SGD([params], lr=learning_rate)

training_loop(model= model,
              n_epochs=5000,
              optimizer=optimizer,
              learning_rate = learning_rate,
              params = params,
              t_u = t_un,
              t_c = t_c)


# ## Respuestas

# Dentro del training loop lo que cambió fue lo que estaba recibiendo como `model`: se le ajusto al nuevo modelo provisto en las instrucciones. Así que el verdadero cambio fue redefinir qué era la variable `model`.

# Luego, todo el training loop quedó exactamente igual.

# El loss resulta ser un poco más alto que el modelo lineal. Por lo que podría concluirse el resultado es menos mejor que el modelo lineal.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




