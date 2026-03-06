#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:56:19 2026

@author: henrysilverman
"""

import pandas as pd
import numpy as np
from openai import OpenAI
import faiss
import time
from openai import APIConnectionError, RateLimitError
client = OpenAI(api_key = "insert_your_api_key_here")

#Reading in  IMDB dataset
imdb = pd.read_csv('/Users/henrysilverman/Downloads/IMDB_Dataset.csv')
imdb = imdb.reset_index(drop=True)

#Truncating long reviews to avoid token limits
def truncate(text, max_chars=1000):
    return text[:max_chars]

#Combining review + sentiment
documents = [
    f"Review: {truncate(row['review'])}\nSentiment: {row['sentiment']}"
    for _, row in imdb.iterrows()
]

#Creating embeddings from text
def get_embeddings(texts, batch_size=100, max_retries=5):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        retries = 0

        while True:
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                break  # success
            except RateLimitError:
                print("Rate limit hit. Sleeping 1 sec...")
                time.sleep(1)
            except APIConnectionError:
                retries += 1
                if retries > max_retries:
                    raise
                print(f"Connection error. Retry {retries}/{max_retries} in 2 sec...")
                time.sleep(2)

        batch_embeddings = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"Processed {i + len(batch)} / {len(texts)}")

        # Optional: save progress every 1000
        if (i + len(batch)) % 1000 == 0:
            np.save("embeddings_progress.npy", np.array(all_embeddings))

    return np.array(all_embeddings)

#Generating embeddings
doc_embeddings = get_embeddings(documents, batch_size=100)
doc_embeddings = np.array(doc_embeddings, dtype='float32')

#Saving and loading embeddings
np.save("imdb_doc_embeddings.npy", doc_embeddings)
doc_embeddings = np.load("imdb_doc_embeddings.npy").astype('float32')

#FAISS index
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
faiss.normalize_L2(doc_embeddings)
index.add(doc_embeddings)

#Semantic Search Function
def semantic_search_faiss(query, documents, index, embeddings, top_n=5):
    # Embed query
    query_emb = np.array(client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding, dtype='float32')
    
    # Normalize
    faiss.normalize_L2(query_emb.reshape(1, -1))
    
    # Search top N
    D, I = index.search(query_emb.reshape(1, -1), top_n)
    
    results = [(documents[i], float(D[0][k])) for k, i in enumerate(I[0])]
    return results

#RAG Function with JSON output
def rag_answer(query, documents, index, embeddings, top_n=5):
    results = semantic_search_faiss(query, documents, index, embeddings, top_n)
    
    # Combine context
    context = "\n\n".join([doc for doc, score in results])
    
    prompt = f"""
You are a data scientist analyzing movie reviews.

Context:
{context}

Question: {query}

Answer in JSON with fields:
- answer: concise answer
- reasoning: brief explanation
"""
    
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )
    
    return response.output_text

#Testing query
query = "What are the most common reasons for negative movie reviews?"
answer = rag_answer(query, documents, index, doc_embeddings, top_n=5)

print("Answer:")
print(answer)