#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch

def evaluation(key_ref, corpus_scores, query_labels, dataset_name):
    recall_threshold = [1, 5, 10]
    recall_results = [0] * len(recall_threshold)
    
    dataset_parts = {
        "perspectrum": ["support", "undermine", "general"],
        "agnews": ["subtopic", "location"],
        "story": ["analogy", "entity"],
        "ambigqa": ["perspective"],
        "allsides": ["left", "right", "center"],
        "exfever": ["SUPPORT", "REFUTE", "NOT ENOUGH INFO"]
    }
    
    parts = dataset_parts.get(dataset_name, ["none"])
    parts_size = [0] * len(parts)
    
    for lb in query_labels:
        parts_size[parts.index(lb)] += 1
            
    partial_recall_results = [[0] * len(recall_threshold) for _ in range(len(parts))]
    
    for k, v in key_ref.items():
        for j, thresh in enumerate(recall_threshold):
            ranked_scores = (-np.array(corpus_scores[int(k)])).argsort()[:thresh]
            indicator = any(int(index) in ranked_scores for index in (v if isinstance(v, list) else [v]))
            recall_results[j] += indicator
            partial_recall_results[parts.index(query_labels[int(k)])][j] += indicator
    
    final_results = [result / len(key_ref) for result in recall_results]
    print("Overall:", "; ".join(f"Recall@{thresh}: {round(final_results[i],3)}" for i, thresh in enumerate(recall_threshold)))

def create_embeddings(tokenizer, model, texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    batch_size = 17
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            batch_embeddings = model(**inputs).pooler_output
            embeddings.extend(batch_embeddings.cpu().tolist())

    return embeddings

