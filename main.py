import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Load and preprocess data

def load_transactions(file_path="transactions.json"):
    with open(file_path, "r") as file:
        data = json.load(file)

    texts = [
        f"On {t['date']}, {t['customer']} purchased a {t['product']} for ₹{t['amount']}."
        for t in data
    ]

    return data, texts


# 2. Create embeddings

def create_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    return model, embeddings



# 3. Similarity-based retriever

def retrieve_transactions(query, model, embeddings, texts, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]
    retrieved = [texts[i] for i in top_indices]

    return retrieved


# 4. Simple rule-based answer generator
# (since we are not using an LLM here)

def generate_answer(question, context):
    answer = "Here are the relevant transactions:\n"
    for c in context:
        answer += f"- {c}\n"
    return answer


# 5. Main chatbot loop

def chatbot():
    print("RAG Chatbot Ready! Ask a question.")
    print("Type 'exit' to stop.\n")

    data, texts = load_transactions()
    model, embeddings = create_embeddings(texts)

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        retrieved = retrieve_transactions(query, model, embeddings, texts)
        answer = generate_answer(query, retrieved)

        print("\nBot:", answer, "\n")


# Run chatbot
if __name__ == "__main__":
    chatbot()
