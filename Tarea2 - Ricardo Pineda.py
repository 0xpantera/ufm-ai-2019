#!/usr/bin/env python
# coding: utf-8

# ## Ejercicios

# Crear un tensor de list(range(9)) e indicar cual es el size, offset, y strides
# Crear un tensor b = a.view(3, 3). Cual es el valor de b[1, 1]
# crear un tensor c = b[1:, 1:]. Cual es el size, offset, strides?
# Escogan una operacion matematica como cosine o sqrt. Hay una funcion correspondiente en PyTorch?
# Existe una version de esa operacion que opera in-place?

# Crear un tensor 2D y luego agregar una dimension de tamanio 1 insertada en la dimension 0.
# Eliminar la dimension extra que agrego en el tensor previo.
# Crear un tensor aleatorio de forma 5𝑥3 en el intervalo [3,7)
# Crear un tensor con valores de una distribucion normal (𝜇=0,𝜎=1)
# Recuperar los indices de todos los elementos no cero en el tensor torch.Tensor([1,1,1,0,1]).
# Crear un tensor aleatorio de forma (3,1) y luego apilar cuatro copias horizontalmente.
# Retornar el producto batch matrix-matrix de dos matrices 3D: (a=torch.randn(3,4,5), b=torch.rand(3,5,4))
# Retornar el producto batch matrix-matrix de una matriz 3D y una matriz 2D: (a=torch.rand(3,4,5), b=torch.rand(5,4)).

# ## Resolucion

# ### Serie 1

# In[1]:


import torch


# In[4]:


a = torch.tensor(list(range(9)))
print('Size de "a" '': ', a.size(), ', Offset:', a.storage_offset(), ', Stride:', a.stride())
b = a.view(3,3)
print('Valor de b[1,1]: ', b[1,1])
c = b[1:, 1:]
print('Size de "c" '': ', c.size(), ', Offset:', c.storage_offset(), ', Stride:', c.stride())


# In[7]:


s1 = torch.ones(3)
s2 = torch.ones(3)
s1.add_(s2)
print(s1)


# ### Serie 2

# In[8]:


a = torch.ones([2, 2])
a.unsqueeze_(0)
print(a.size())


# In[9]:


a.resize_((2, 2))
print(a.size())


# In[10]:


a = torch.randint(size = [5, 3], low = 3, high = 7)
print(a)


# In[11]:


a = torch.normal(torch.zeros(4), torch.ones(4))
print(a)


# In[12]:


a = torch.Tensor([1, 1, 1, 0, 1])
print((a == 0).nonzero()[0])


# In[13]:


a = torch.rand(3, 1)
b = torch.stack([a, a, a, a])
print(b.size())


# In[ ]:


a = torch.rand(3, 4, 5)
b = torch.rand(5, 4)
producto = torch.bmm(a, b.expand(3, 5, 4))
print(producto)


# In[ ]:




