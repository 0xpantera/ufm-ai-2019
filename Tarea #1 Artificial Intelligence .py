#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch


# In[12]:


print('float 16')
print('Max')
print(torch.finfo(torch.float16).max)
print('Min')
print(torch.finfo(torch.float16).min)
print('float 32')
print('Max')
print(torch.finfo(torch.float32).max)
print('Min')
print(torch.finfo(torch.float32).min)
print('float 64')
print('Max')
print(torch.finfo(torch.float64).max)
print('Min')
print(torch.finfo(torch.float64).min)
print('integer uns. 8')
print('Max')
print(torch.iinfo(torch.uint8).max)
print('Min')
print(torch.iinfo(torch.uint8).min)
print('integer s. 8')
print('Max')
print(torch.iinfo(torch.int8).max)
print('Min')
print(torch.iinfo(torch.int8).min)
print('integer s. 16')
print('Max')
print(torch.iinfo(torch.int16).max)
print('Min')
print(torch.iinfo(torch.int16).min)
print('integer s. 32')
print('Max')
print(torch.iinfo(torch.int32).max)
print('Min')
print(torch.iinfo(torch.int32).min)
print('integer s. 64')
print('Max')
print(torch.iinfo(torch.int64).max)
print('Min')
print(torch.iinfo(torch.int64).min)


# In[14]:


print('Como se puede ver, los valores maximos para cada tipo de dato se encuentran en el resultado anterior.')


# In[15]:


print('La diferencia entre un int signed y unsigned es que los ints que son denominados signed pueden tomar valores negativos o positivos, mientras que los unsigned solo pueden tomar valores positivos, incluyendo 0. Tambien es importante notar que ambos tipos de datos tienen la misma cantidad de valores que pueden tomar, solo que, como se menciono anteriormente, en los signed, los valores pueden ser negativos y positivos, mientras que en los unsigned pueden ser unicamente positivos.')


# In[27]:


print('Los tipos de datos default para python base, numpy y pytorch son los siguientes: para base python, dependiendo del input, el default puede ser int (si el numero ingresado es un entero), float(si el numero tiene decimales), o complex(si el numero contiene aspectos imaginarios), para Numpy el data type default es un float de 64 bits, y por último, el data type default de pytorch es un float de 32 bits.')

