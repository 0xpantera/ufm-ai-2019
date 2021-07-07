#!/usr/bin/env python
# coding: utf-8

# # Ricardo Pineda | 20160164

# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

print("Paquetes importados con exito")


# In[4]:


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c).unsqueeze(1)
t_u = torch.tensor(t_u).unsqueeze(1)

n_samples = t_u.shape[0]
n_val = int(0.2 / n_samples)


# In[5]:


shuffled_indices = torch.randperm(n_samples)


# In[6]:


train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]


# In[7]:


train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u


# In[8]:


class SubclassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(1, 13)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(13, 1)


    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)

        return output_t


# In[9]:


def training_loop(model, n_epochs, optimizer, loss_fn, train_x, val_x, train_y, val_y):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_x) # ya no tenemos que pasar los params
        train_loss = loss_fn(train_t_p, train_y)

        with torch.no_grad(): # todos los args requires_grad=False
            val_t_p = model(val_x)
            val_loss = loss_fn(val_t_p, val_y)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss}, Validation loss {val_loss}")


# In[10]:


subclass_model = SubclassModel()
optimizer = optim.SGD(subclass_model.parameters(), lr=1e-3)


# In[14]:


def training_loop(model, n_epochs, optimizer, loss_fn, train_x, val_x, train_y, val_y):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_x)
        train_loss = loss_fn(train_t_p, train_y)

        with torch.no_grad(): 
            val_t_p = model(val_x)
            val_loss = loss_fn(val_t_p, val_y)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss}, Validation loss {val_loss}")


# In[18]:


class RedNeuronal(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(1, 100)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(100, 1)


    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)

        return output_t
subclass_model = RedNeuronal()

optimizer = optim.SGD(subclass_model.parameters(), lr=1e-9)  


# In[21]:


from numpy import genfromtxt
import numpy as np
data = genfromtxt('winequality-white.csv', delimiter=';')
data = np.array(data[1:])
data = torch.from_numpy(data).float()
features = data[:,:-1]
target = data[:,-1].unsqueeze(1)


# In[22]:


class WineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(11, 10)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(10, 1)


    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)

        return output_t


# In[23]:


validation_losses = []
epochs_losses = []
training_losses = []


# In[24]:


def training_loop(model, n_epochs, optimizer, loss_fn, train_x, val_x, train_y, val_y):
    print(train_x.shape)
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_x) # ya no tenemos que pasar los params
        train_loss = loss_fn(train_t_p, train_y)
        with torch.no_grad(): # todos los args requires_grad=False
            val_t_p = model(val_x)
            val_loss = loss_fn(val_t_p, val_y)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss}, Validation loss {val_loss}")

        validation_losses.append(val_loss)
        epochs_losses.append(epoch)
        training_losses.append(train_loss)


# In[25]:


wine_model = WineModel()

t_c = target
t_u = features

n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = train_t_u
val_t_un = val_t_u

optimizer = optim.SGD(wine_model.parameters(), lr=1e-3)


# In[26]:


print(train_t_un.shape,val_t_un.shape,train_t_c.shape,val_t_c.shape)


# In[27]:


plt.plot(epochs_losses, training_losses)
plt.title('Training loss during epochs')
plt.show()


# In[28]:


plt.plot(epochs_losses, validation_losses)
plt.title('Validation loss during epochs')


# ### Preguntas y Respuestas

# Experimenten con el numero de neuronas en el modelo al igual que el learning rate.
# 
# - Que cambios resultan en un output mas lineal del modelo?
#     - El loss disminuye
# 
# - Pueden hacer que el modelo haga un overfit obvio de la data?
#     - Poniendo un Learning Rate bajo
# 
# - Cargen la data de vinos blancos y creen un modelo con el numero apropiado de inputs
# 
# - Cuanto tarda en entrenar comparado al dataset que hemos estado usando?
#     - Se tarda mas o menos el doble
# 
# - Pueden explicar que factores contribuyen a los tiempos de entrenamiento?
#     - Numero de Neuronas
#     - Cantidad de Layers
#     
# - Pueden hacer que el loss disminuya?
#     - No realmente pues se daria overfitting
# 
# - Intenten graficar la data

# In[ ]:




