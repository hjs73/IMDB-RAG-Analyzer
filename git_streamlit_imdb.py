#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:58:40 2026

@author: henrysilverman
"""

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
client = OpenAI(api_key = "insert_your_api_key_here")

#Loading in Documents and Embeddings
st.title("IMDB Review RAG Analyzer")

@st.cache_data
def load_data():
    # Load dataset
    imdb = pd.read_csv('/Users/henrysilverman/Downloads/IMDB_Dataset.csv')
    imdb = imdb.reset_index(drop=True)

    # Truncate long reviews
    def truncate(text, max_chars=1000):
        return text[:max_chars]

    # Combine review + sentiment
    documents = [
        f"Review: {truncate(row['review'])}\nSentiment: {row['sentiment']}"
        for _, row in imdb.iterrows()
    ]

    # Load precomputed embeddings
    embeddings = np.load("imdb_doc_embeddings.npy").astype('float32')
    return documents, embeddings

documents, doc_embeddings = load_data()

#Building FAISS index
@st.cache_data
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Cosine similarity with normalized vectors
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

index = build_index(doc_embeddings)

#Semantic Search
def semantic_search_faiss(query, documents, index, embeddings, top_n=5):
    query_emb = np.array(
        client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        ).data[0].embedding, dtype='float32'
    )
    faiss.normalize_L2(query_emb.reshape(1, -1))
    D, I = index.search(query_emb.reshape(1, -1), top_n)
    results = [(documents[i], float(D[0][k])) for k, i in enumerate(I[0])]
    return results

#RAG function
def rag_answer(query, documents, index, embeddings, top_n=5):
    results = semantic_search_faiss(query, documents, index, embeddings, top_n)
    context = "\n\n".join([doc for doc, score in results])

    prompt = f"""
You are a data scientist analyzing movie reviews.

Context:
{context}

Question: {query}

Answer concisely.
"""
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )
    return response.output_text

#Streamlit UI
query = st.text_input("Ask a question about the IMDB reviews, e.g., 'Which comedy movies have the most positive reviews?' or 'Summarize reviews for The Godfather:'", "")

if query:
    with st.spinner("Generating answer..."):
        answer = rag_answer(query, documents, index, doc_embeddings)
    st.subheader("Answer:")
    st.write(answer)