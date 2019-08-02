#!/usr/bin/env python
# coding: utf-8

# In[65]:


import torch


# **1. Crear un tensor de list(range(9)) e indicar cual es el size, offset, y strides**

# In[69]:


a = torch.tensor(list(range(9)))
print(a)

print("Size: ", a.size())
print("Offset: ", a.storage_offset())
print("Stride: ", a.stride())


# - Crear un tensor b = a.view(3, 3). Cual es el valor de b[1, 1]

# In[70]:


b = a.view(3,3)
print(b)
print()
print("b[1,1] =", b[1,1])


# - Crear un tensor c = b[1:, 1:]. Cual es el size, offset, strides?

# In[71]:


c = b[1:,1:]
print(c)
print()

size = c.size()
offset = c.storage_offset()
stride = c.stride()

print("Size: "f'{size}')
print("Offset: "f'{offset}')
print("Stride: "f'{stride}')


# **2. Escojan una operacion matematica como cosine o sqrt. Hay una funcion correspondiente en PyTorch?**

# In[72]:


d = torch.FloatTensor([1.0, -0.5, 3.4, -2.1, 0.0, -6.5]) 
print(d)
print()
print("funcion torch.cos()")
print(torch.cos(d))


# - Existe una version de esa operacion que opera in-place?

# In[73]:


import math

e = [1.0, -0.5, 3.4, -2.1, 0.0, -6.5]
for i in e:
    print(math.cos(i))


# `in-place` per se no hay, se puede conseguir mediante la libreria `math`

# **3. Crear un tensor 2D y luego agregar una dimension de tamanio 1 insertada en la dimension 0.**

# In[74]:


x1 = torch.zeros(5, 5)
print(x1)
print(x1.size())
print()

print(x1.unsqueeze(0))
print(x1.unsqueeze(0).size())


# **4. Eliminar la dimension extra que agrego en el tensor previo**

# In[75]:


print(x1.squeeze())
print(x1.squeeze().size())


# **5. Crear un tensor aleatorio de forma $5x3$ en el intervalo $[3,7)$**

# In[76]:


rand_tensor = torch.randint(3,7,(5,3))
rand_tensor


# **6. Crear un tensor con valores de una distribucion normal ($\mu=0, \sigma=1$)**

# In[77]:


norm_tensor = torch.distributions.normal.Normal(0, 1)
norm_tensor.sample()


# **7. Recuperar los indices de todos los elementos no cero en el tensor torch.Tensor([1,1,1,0,1]).**

# In[78]:


torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))


# **8. Crear un tensor aleatorio de forma (3,1) y luego apilar cuatro copias horizontalmente.**

# In[79]:


rand1 = torch.randn(3, 1)
print(rand1)

rand2 = rand1.clone()
print(rand2)

rand3 = rand1.clone()
print(rand3)

rand4 = rand1.clone()
print(rand4)

rand5 = rand1.clone()
print(rand5)


# In[80]:


tensor_list = [rand1, rand2, rand3, rand4, rand5]
print(tensor_list)


# In[81]:


stacked_tensor = torch.stack(tensor_list)
print(stacked_tensor)


# **7. Retornar el producto batch matrix-matrix de dos matrices 3D: (a=torch.randn(3,4,5), b=torch.rand(3,5,4))**

# In[82]:


a=torch.randn(3,4,5)
print(a)
print()
b=torch.rand(3,5,4)
print(b)


# In[83]:


torch.bmm(a, b)


# In[84]:


torch.matmul(a,b)


# `matmul` y `bmm` devuelven el mismo resultado en este caso

# **8. Retornar el producto batch matrix-matrix de una matriz 3D y una matriz 2D: (a=torch.rand(3,4,5), b=torch.rand(5,4)).**

# In[85]:


c =torch.rand(3,4,5)
print(a)
d =torch.rand(5,4)
print()
print(b)


# In[86]:


torch.matmul(c,d)

