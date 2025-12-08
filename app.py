import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# 1. Load + Preprocess Data

@st.cache_resource
def load_transactions(file_path="transactions.json"):
    with open(file_path, "r") as file:
        data = json.load(file)

    # Convert each transaction into text
    texts = [
        f"Transaction ID {t['id']}: On {t['date']}, {t['customer']} purchased a {t['product']} for ₹{t['amount']}."
        for t in data
    ]

    return data, texts


# 2. Create Embeddings

@st.cache_resource
def create_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    return model, embeddings


# 3. Retriever


def retrieve_transactions(query, model, embeddings, texts, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # top-k most similar
    top_indices = similarities.argsort()[::-1][:top_k]
    retrieved = [texts[i] for i in top_indices]

    return retrieved


# 4. LLM-Based Answer Generator (Real RAG)

def generate_answer(question, context_list):
    context = "\n".join(context_list)

    prompt = f"""
You are a strict RAG chatbot.
Answer the user's question using ONLY this context.
If answer is not found in the context, say:
"Information not available in the provided transactions."

Context:
{context}

Question:
{question}

If needed, calculate totals, counts or averages only from this context.
Give a clear, helpful answer.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message["content"]


# Streamlit UI

st.title(" RAG Chatbot for Transactions")
st.write("Ask any question about customer purchases based on the dataset!")

# Load data + embeddings
data, texts = load_transactions()
model, embeddings = create_embeddings(texts)

# User input
query = st.text_input("Enter your question:")

if query:
    retrieved = retrieve_transactions(query, model, embeddings, texts)
    answer = generate_answer(query, retrieved)

    st.subheader("Retrieved Context")
    for r in retrieved:
        st.write("- ", r)

    st.subheader("Chatbot Answer")
    st.write(answer)
