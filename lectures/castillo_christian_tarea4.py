#!/usr/bin/env python
# coding: utf-8

# In[153]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# In[154]:


vinos = np.loadtxt("vinos.csv", delimiter = ";", skiprows = 1)


# In[155]:


vinos


# In[156]:


x = vinos[:,0:11]
x


# In[157]:


y = vinos[:,-1]
y


# In[158]:


x = torch.from_numpy(x)
x


# In[159]:


y = torch.from_numpy(y)
y


# In[160]:


y = y.unsqueeze(1)
y


# In[161]:


print(x.size())
print(y.size())


# In[170]:


x = x.float()
y = y.float()


# In[176]:


n_samples = x.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_x = x[train_indices]
train_y = y[train_indices]

val_x = x[val_indices]
val_y = y[val_indices]

train_x_n = 0.1 * train_x
val_x_n = 0.1 * val_x


# In[178]:


def training_loop(model, n_epochs, optimizer, loss_fn, train_x, val_x, train_y, val_y):
    for epoch in range(1, n_epochs + 1):
        train_y_p = model(train_x_n) 
        train_loss = loss_fn(train_y_p, train_y)
        
        with torch.no_grad(): 
            val_y_p = model(val_x_n)
            val_loss = loss_fn(val_y_p, val_y)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss}, Validation loss {val_loss}")


# In[179]:


class SubclassFunctionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(11, 9)
        self.output_linear = nn.Linear(9, 11)

        
    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = torch.tanh(hidden_t)
        output_t = self.output_linear(activated_t)

        return output_t


func_model = SubclassFunctionalModel()
func_model


# In[182]:


optimizer = optim.SGD(func_model.parameters(), lr=1e-4)

training_loop(
    n_epochs=5000,
    optimizer=optimizer,
    model=func_model,
    loss_fn=nn.MSELoss(),
    train_x = train_x_n,
    val_x = val_x_n,
    train_y = train_y,
    val_y = val_y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




