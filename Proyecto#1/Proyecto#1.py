#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import sys
import ai_utils


# In[2]:


with open('together.txt', 'r',  encoding="utf8") as f:
    text = f.read()
fname = sys.argv[1]
 
num_words = 0
 
with open('together.txt', 'r',  encoding="utf8") as f:
    for line in f:
        words = line.split()
        num_words += len(words)
print("---------------------------------------------------------------------------------------------------------------")
print("Primeros 100 caracteres:")
print(text[:100])
print("\nNumero de Palabras:")
print(num_words)
print("---------------------------------------------------------------------------------------------------------------")


# In[3]:


chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
encoded = np.array([char2int[ch] for ch in text])
encoded[:100]
len(encoded)


# In[4]:


n_hidden=512
n_layers=3

net = ai_utils.CharRNN(chars, n_hidden, n_layers)
print(net)

batch_size = 128
seq_length = 150
n_epochs = 22
print("---------------------------------------------------------------------------------------------------------------")


# In[5]:



    

                
ai_utils.train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=50, val_frac=0.15)

model_name = 'rnn_20_epoch.net'

checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)


# In[7]:


print("---------------------------------------------------------------------------------------------------------------")
print(ai_utils.sample(net, 2000, prime='Es', top_k=8))


# In[ ]:




