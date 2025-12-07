import streamlit as st
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Load Data
@st.cache_resource
def load_data():
    with open("transactions.json", "r") as f:
        data = json.load(f)

    texts = [
        f"On {t['date']}, {t['customer']} purchased a {t['product']} for ₹{t['amount']}."
        for t in data
    ]
    return data, texts


# 2. Embeddings

@st.cache_resource
def load_model_and_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    return model, embeddings



# 3. RAG Retriever with Customer Filtering

def retrieve(query, model, embeddings, texts, data, top_k=5):

    # Step 1: similarity search 
    query_embed = model.encode([query])
    sims = cosine_similarity(query_embed, embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    retrieved = [(texts[i], data[i]) for i in top_idx]

    # Step 2: detect customer name from query 
    customers = ["Amit", "Riya", "Karan"]
    name_in_query = None
    for c in customers:
        if c.lower() in query.lower():
            name_in_query = c

    # Step 3: filter retrieved results by customer name 
    if name_in_query:
        retrieved = [t for t in retrieved if t[1]["customer"] == name_in_query]

    # return only the text strings
    return [t[0] for t in retrieved]

# 4. Streamlit UI

st.title("RAG Chatbot for Transactions")
st.write("Ask any question about customer purchases based on the dataset!")

data, texts = load_data()
model, embeddings = load_model_and_embeddings(texts)

query = st.text_input("Enter your question:")

if query:
    results = retrieve(query, model, embeddings, texts, data)

    st.subheader("Retrieved Transactions")
    if len(results) == 0:
        st.write("No matching transactions found.")
    else:
        for r in results:
            st.write(f"- {r}")

    st.subheader("Chatbot Answer")
    if len(results) > 0:
        st.write("Here are the relevant transactions found:")
        for r in results:
            st.write(f"{r}")
    else:
        st.write("Sorry, I couldn't find any relevant transactions.")
