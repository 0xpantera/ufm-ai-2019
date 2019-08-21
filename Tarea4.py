#!/usr/bin/env python
# coding: utf-8

# Alumno: Carlos Cujcuj

# In[20]:


import torch
import torch.nn as nn
import torch.optim as optim


# In[21]:



t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] # Temperatura en grados celsios
t_u = [35.7, 55.9, 58.2,81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] # Unidades desconocidas
t_c = torch.tensor(t_c).unsqueeze(1) # Agregamos una dimension para tener B x N_inputs
t_u = torch.tensor(t_u).unsqueeze(1) # Agregamos una dimension para tener B x N_inputs

n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u


# In[22]:


class SubclassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(1, 13)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(13, 1)

        
    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        #activated_t = self.hidden_activation(hidden_t) if random.random() > 0.5 else hidden_t
        output_t = self.output_linear(activated_t)

        return output_t
    
    


# In[23]:


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


# In[24]:


subclass_model = SubclassModel()
optimizer = optim.SGD(subclass_model.parameters(), lr=1e-3)


# In[25]:


training_loop(
    n_epochs=3000,
    optimizer=optimizer,
    model=subclass_model,
    loss_fn=nn.MSELoss(), # Ya no estamos usando nuestra loss function hecha a mano
    train_x = train_t_un,
    val_x = val_t_un,
    train_y = train_t_c,
    val_y = val_t_c)


# **Que cambios resultan en un output mas lineal del modelo?**

# Cambiando el *learning rate* del modelo y los epocs que realizara

# **Pueden hacer que el modelo haga un overfit obvio de la data?**

# Un numero menor al learning rate ya que haria pasos muy pequeños en el gradiente. O haciendo lo contrario 

# ## Cargen la data de vinos blancos y creen un modelo con el numero apropiado de inputs

# In[72]:


from numpy import genfromtxt
import numpy as np
data = genfromtxt('winequality-white.csv', delimiter=';')


# In[27]:


data.shape


# In[61]:


wines.shape


# In[62]:


type(wines)


# In[80]:


wines = np.array(wines[1:])

wines = torch.from_numpy(wines).float()
feat = wines[:,:-1]
targ = wines[:,-1].unsqueeze(1)


# In[81]:


class ModeloDeVinos(nn.Module):
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
    


# In[82]:


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


# In[83]:


n_samples = feat.shape[0]
n_val = int(0.2 * n_samples) # 20% del dataset original sera de validacion 

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_feat = feat[train_indices]
train_targ = targ[train_indices]

val_feat = feat[val_indices]
val_targ = targ[val_indices]


# In[84]:


red = ModeloDeVinos()
optimizador = optim.SGD(red.parameters(), lr=1e-3)


# In[86]:


from timeit import default_timer as timer

start = timer()

training_loop(
    n_epochs=3000,
    optimizer=optimizador,
    model=red,
    loss_fn=nn.MSELoss(), # Ya no estamos usando nuestra loss function hecha a mano,
    train_x = train_feat,
    val_x = val_feat,
    train_y = train_targ,
    val_y = val_targ)

end = timer()
total = end - start


# **Cuanto tarda en entrenar comparado al dataset que hemos estado usando?**

# In[87]:


print("Tiempo total en seg: ", total)


# **Pueden explicar que factores contribuyen a los tiempos de entrenamiento?**

# El tamaño del learning rate afecta ya que depende que tan grande o pequeño es el paso que dara.   
# Tambien el numero de capas y neuronas que se especifican

# **Pueden hacer que el loss disminuya?**

# Se puede llegar a un punto aceptable pero si se intenta llegar a un valor muy cercano a cero a 'fuerza', podria presentar un overfitting

# In[ ]:




