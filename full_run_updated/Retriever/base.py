import json
import torch
import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm
from abc import ABC, abstractmethod


class BaseRetriever:
    def __init__(self, model_name):
        self.model_name = model_name

        # Load tokenizer & model dynamically
        model_data = self.load_model()

        # Handle models with one tokenizer (e.g., Contriever) or two (e.g., DPR)
        if isinstance(model_data[0], tuple):  
            # If `load_model` returns ((tokenizer1, model1), (tokenizer2, model2))
            (self.ctokenizer, self.cmodel), (self.qtokenizer, self.qmodel) = model_data
        else:
            # If `load_model` returns (tokenizer, model)
            self.tokenizer, self.model = model_data
            self.cmodel = self.model  # For consistency in naming
            self.ctokenizer = self.tokenizer

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move models to the correct device
        if hasattr(self, "cmodel"):
            self.cmodel.to(self.device)
            self.cmodel.eval()

        if hasattr(self, "qmodel"):
            self.qmodel.to(self.device)
            self.qmodel.eval()

# class BaseRetriever(ABC):
#     def __init__(self, model_name):
#         self.model_name = model_name
#         self.tokenizer, self.model = self.load_model()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#         self.model.eval()

#     @abstractmethod
#     def load_model(self):
#         """Subclasses must implement model loading."""
#         pass

    @abstractmethod
    def embed_texts(self, texts, text_type):
        """Subclasses must implement embedding generation."""
        pass

    def retrieve(self, datasets):
        """Common retrieval workflow."""
        for k, v in datasets.items():
            print(f"Processing dataset: {k}")
            queries, corpus = v["queries"], v["corpus"]
            key_ref, query_labels = v["key_ref"], v["query_labels"]

            query_embeddings = self.embed_texts(queries, 'queries')
            corpus_embeddings = self.embed_texts(corpus, 'corpus')

            corpus_scores = self.compute_similarity(query_embeddings, corpus_embeddings)
            self.save_scores(k, corpus_scores)
            from Retriever.utils import evaluation
            evaluation(key_ref, corpus_scores, query_labels, k)

    def compute_similarity(self, query_embeddings, corpus_embeddings):
        """Compute cosine similarity between query and corpus embeddings."""
        return [
            [1 - cosine(emb1, emb2) for emb2 in corpus_embeddings]
            for emb1 in tqdm(query_embeddings)
        ]

    def save_scores(self, dataset_name, scores):
        """Save similarity scores to a JSON file."""
        filename = f"{self.model_name}_{dataset_name}_scores.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(scores, f)



# #!/usr/bin/env python
# # coding: utf-8

# # In[ ]:


# import json
# import torch
# import numpy as np
# from scipy.spatial.distance import cosine
# from tqdm import tqdm
# from abc import ABC, abstractmethod

# class BaseRetriever(ABC):
#     def __init__(self, model_name):
#         self.model_name = model_name
#         self.tokenizer, self.model = self.load_model()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#         self.model.eval()

#     @abstractmethod
#     def load_model(self):
#         """Subclasses must implement model loading."""
#         pass

#     @abstractmethod
#     def embed_texts(self, texts):
#         """Subclasses must implement embedding generation."""
#         pass

#     def retrieve(self, datasets):
#         """Common retrieval workflow."""
#         for k, v in datasets.items():
#             print(f"Processing dataset: {k}")
#             queries, corpus = v["queries"], v["corpus"]
#             key_ref, query_labels = v["key_ref"], v["query_labels"]

#             query_embeddings = self.embed_texts(queries)
#             corpus_embeddings = self.embed_texts(corpus)

#             corpus_scores = self.compute_similarity(query_embeddings, corpus_embeddings)
#             self.save_scores(k, corpus_scores)
#             from Retriever.utils import evaluation
#             evaluation(key_ref, corpus_scores, query_labels, k)

#     def compute_similarity(self, query_embeddings, corpus_embeddings):
#         """Compute cosine similarity between query and corpus embeddings."""
#         return [
#             [1 - cosine(emb1, emb2) for emb2 in corpus_embeddings]
#             for emb1 in tqdm(query_embeddings)
#         ]

#     def save_scores(self, dataset_name, scores):
#         """Save similarity scores to a JSON file."""
#         filename = f"{self.model_name}_{dataset_name}_scores.json"
#         with open(filename, "w", encoding="utf-8") as f:
#             json.dump(scores, f)

