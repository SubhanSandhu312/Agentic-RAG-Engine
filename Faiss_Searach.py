#!/usr/bin/env python
# coding: utf-8

# In[2]:


import faiss
import numpy as np
from embeddings import embeddings_list


# In[3]:


embeddings_array = np.array(embeddings_list, dtype='float32')


# print(embeddings_array.shape)


# In[4]:


dimension = 384

index = faiss.IndexFlatL2(dimension)

index.add(embeddings_array)


# In[5]:


# query_vector = np.random.rand(1, dimension).astype('float32')
def query_indices(query_vector, top_k=3):
    scores, indices = index.search(query_vector, top_k)
    return scores, indices
# top_k = 3
# scores, indices = index.search(query_vector, top_k)

# print("Indices of nearest chunks:", indices)
# print("L2 Distances:", scores)

