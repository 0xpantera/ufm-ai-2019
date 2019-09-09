#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[7]:


#Ejercicio 1
a = torch.Tensor(list(range(9)))
print ("El tamaño de a es de", a.size())
print ("El stride de a es de ", a.stride())
print ("El offset de a es de ", a.storage_offset())
b = a.view(3,3)
print ("El valor de b[1, 1] es de ", b[1, 1])
c = b[1:, 1:]
print ("El tamaño de c es de", c.size())
print ("El stride de c es de ", c.stride())
print ("El offset de c es de ", c.storage_offset())


# In[6]:


#Ejercicio 2
torchcos = torch.Tensor([0,1,2])
torchcos_1 = torch.cos(torchcos)
print(torchcos_1)
#In place
torch.cos_(torch.Tensor([0,1,2]))


# In[20]:


#Ejercicio 3
d = torch.Tensor([[0,1,2],[0,1,2]])
d.unsqueeze_(0).shape


# In[15]:


#Ejercicio 4
d.squeeze_(0).shape


# In[2]:


#Ejercicio 5
aleatorios = torch.randint(low = 3, high = 7, size = (5,3))
aleatorios


# In[21]:


#Ejercicio 6
dist_normal = torch.randn(3,3)
dist_normal


# In[23]:


#Ejercicio 7
nz = torch.Tensor([1,1,1,0,1])
index = torch.nonzero(nz)
index


# In[30]:


#Ejercicio 8
aleatorios = torch.rand(3,1).t()
tensor_final = torch.cat([aleatorios, aleatorios, aleatorios, aleatorios], dim = 0).t()
tensor_final


# In[24]:


#Ejercicio 9
a = torch.randn(3,4,5)
b = torch.rand(3,5,4)
c = torch.matmul(a,b)
c


# In[25]:


#Ejercicio 10
a = torch.randn(3,4,5)
b = torch.rand(5,4)
c = torch.matmul(a,b)
c

