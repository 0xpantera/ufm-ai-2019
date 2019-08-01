#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

