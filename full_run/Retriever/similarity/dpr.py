#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    AutoTokenizer, AutoModel
)
from Retriever.base import BaseRetriever
from Retriever.utils import create_embeddings

class DPRRetriever(BaseRetriever):
    def load_model(self):
        """Loads DPR or SimCSE models."""
        if self.model_name == "dpr":
            ctokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            cmodel = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            qtokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            qmodel = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        else:  # SimCSE models
            model_mapping = {
                "simcse-unsup": "princeton-nlp/unsup-simcse-bert-base-uncased",
                "simcse-sup": "princeton-nlp/sup-simcse-bert-base-uncased",
            }
            ctokenizer = AutoTokenizer.from_pretrained(model_mapping[self.model_name])
            cmodel = AutoModel.from_pretrained(model_mapping[self.model_name])
            qtokenizer = AutoTokenizer.from_pretrained(model_mapping[self.model_name])
            qmodel = AutoModel.from_pretrained(model_mapping[self.model_name])

        return (ctokenizer, cmodel), (qtokenizer, qmodel)

    def embed_texts(self, texts):
        """Generates embeddings using DPR/SimCSE models."""
        return create_embeddings(self.tokenizer, self.model, texts)

