#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import AutoTokenizer, AutoModel
import torch
from Retriever.base import BaseRetriever
from Retriever.utils import create_embeddings

class ContrieverRetriever(BaseRetriever):
    def load_model(self):
        """Loads the Contriever model."""
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        model = AutoModel.from_pretrained("facebook/contriever")
        return tokenizer, model

    def embed_texts(self, texts):
        """Generate embeddings using mean pooling."""
        return create_embeddings(self.tokenizer, self.model, texts)

