#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# 1. Crear un tensor de list(range(9)) e indicar cual es el size, offset, y strides
#  - Crear un tensor b = a.view(3, 3). Cual es el valor de b[1, 1]
#  - crear un tensor c = b[1:, 1:]. Cual es el size, offset, strides?

# In[2]:


a =torch.tensor(list(range(9)))
print(a,"\n",
      a.size(), "\n",
      a.storage_offset(), "\n",
      a.stride(), "\n")

b = a.view(3,3)
print(b,"\n",
      'valor de b[1,1]: ', b[1,1], "\n")

c = b[1:, 1:]
print(c, "\n",
      c.size(), "\n",
      c.storage_offset(), "\n",
      c.stride())


# 2. Escogan una operacion matematica como cosine o sqrt. Hay una funcion correspondiente en PyTorch?
# 
# R// Si hay operaciones correspondientes, como `torch.sqrt()`
#  - Existe una version de esa operacion que opera in-place?
#  Si las hay `torch.sqrt_()` por ejemplo

# In[3]:


t1= torch.tensor([[4],
                  [9]], dtype=torch.float)
t2=torch.sqrt(t1)
print(t1, "\n",
t2)


# 1. Crear un tensor 2D y luego agregar una dimension de tamanio 1 insertada en la dimension 0.

# In[4]:


a= torch.ones(2,3)
#a.size()
a= a.unsqueeze_(0)
a.size()


# 2. Eliminar la dimension extra que agrego en el tensor previo.

# In[5]:


a= a.resize_(2,3)
a.size()


# 3. Crear un tensor aleatorio de forma $5x3$ en el intervalo $[3,7)$

# In[6]:


b=torch.randint(low=3, high=7, size=(5,3))
b


# 4. Crear un tensor con valores de una distribucion normal ($\mu=0, \sigma=1$)

# In[7]:


d= torch.normal(mean= 0, std=torch.ones(2,3))
d


# 5. Recuperar los indices de todos los elementos no cero en el tensor `torch.Tensor([1,1,1,0,1])`.

# In[8]:


e=torch.tensor([1,1,1,0,1])
e.nonzero()


# 6. Crear un tensor aleatorio de forma `(3,1)` y luego apilar cuatro copias horizontalmente.

# In[9]:


f = torch.rand(3, 1)
f = torch.stack([f, f, f, f])
f


# 7. Retornar el producto batch matrix-matrix de dos matrices 3D: (`a=torch.randn(3,4,5)`, `b=torch.rand(3,5,4)`)

# In[10]:


a=torch.rand(3,4,5)
b=torch.rand(3,5,4)
g= torch.bmm(a,b)
g


# 8. Retornar el producto batch matrix-matrix de una matriz 3D y una matriz 2D: (`a=torch.rand(3,4,5)`, `b=torch.rand(5,4)`).

# In[11]:


a=torch.rand(3,4,5)
b=torch.rand(5,4)
h= torch.bmm(a, b.expand(3,5,4))
h

