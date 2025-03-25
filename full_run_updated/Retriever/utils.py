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
    if "source" in dataset_name:
        parts = ["none"]
    else:
        parts = dataset_parts.get(dataset_name)
        
    parts_size = [0] * len(parts)
    
    for lb in query_labels:
        parts_size[parts.index(lb)] += 1
            
    partial_recall_results = [[0] * len(recall_threshold) for _ in range(len(parts))]

    for k,v in key_ref.items():
        for j, thresh in enumerate(recall_threshold):
            # important: find one is ok, this can be modified
            ranked_scores = (-np.array(corpus_scores[int(k)])).argsort()[:thresh]
            
            indicator = 0
            try:
                for index in v:
                    if int(index) in ranked_scores:
                        indicator = 1 
            except:
                for index in [v]:
                    if int(index) in ranked_scores:
                        indicator = 1                
            recall_results[j] += indicator
            partial_recall_results[parts.index(query_labels[int(k)])][j] += indicator
    
    final_results = [result/len(key_ref.items()) for result in recall_results]
        
    print("overall",end=": ")
    for i, thresh in enumerate(recall_threshold):
        print("Recall@"+str(thresh)+":",round(final_results[i],3),end="; ")
        
    macro_threshs = [[] for x in recall_threshold]
    print()
    for t, recall_results in enumerate(partial_recall_results):
        print(parts[t],end=": ")
        final_results = [result/parts_size[t] for result in recall_results]
        
        for i, thresh in enumerate(recall_threshold):
            print("Recall@"+str(thresh)+":",round(final_results[i],3),end="; ")
            macro_threshs[i].append(final_results[i])
        print()
    
    # for k, v in key_ref.items():
    #     for j, thresh in enumerate(recall_threshold):
    #         ranked_scores = (-np.array(corpus_scores[int(k)])).argsort()[:thresh]
    #         indicator = any(int(index) in ranked_scores for index in (v if isinstance(v, list) else [v]))
    #         recall_results[j] += indicator
    #         partial_recall_results[parts.index(query_labels[int(k)])][j] += indicator
    
    # final_results = [result / len(key_ref) for result in recall_results]
    # print("Overall:", "; ".join(f"Recall@{thresh}: {final_results[i],3}" for i, thresh in enumerate(recall_threshold)))

# def create_embeddings(tokenizer, model, texts):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
    
#     batch_size = 17
#     embeddings = []

#     for i in range(0, len(texts), batch_size):
#         batch_texts = texts[i:i + batch_size]
#         inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        
#         with torch.no_grad():
#             batch_embeddings = model(**inputs).pooler_output
#             embeddings.extend(batch_embeddings.cpu().tolist())

#     return embeddings

def mean_pooling(model_output, attention_mask):
    """Computes mean pooling over token embeddings.""" # for Contriever
    token_embeddings = model_output.masked_fill(~attention_mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return sentence_embeddings

def create_embeddings(tokenizer, model, texts, method="pooler_output"): # default is pooler output, for DPR
    """
    Tokenizes input texts and extracts embeddings using the specified method.

    Args:
        tokenizer: Tokenizer from Hugging Face.
        model: Hugging Face transformer model.
        texts (List[str]): List of input texts.
        method (str): Either "pooler_output" or "mean_pooling".

    Returns:
        List[List[float]]: List of embeddings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 17
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, max_length=80, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

            if method == "pooler_output":
                batch_embeddings = outputs.pooler_output  # extract CLS token embeddings
            elif method == "mean_pooling":
                batch_embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])  # mean pooling
            else:
                raise ValueError(f"Invalid embedding method: {method}. Choose 'pooler_output' or 'mean_pooling'.")

            embeddings.extend(batch_embeddings.detach().cpu().tolist())

    torch.cuda.empty_cache()
    return embeddings


