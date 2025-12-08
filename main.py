import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# 1. Load + Preprocess Data


def load_transactions(file_path="transactions.json"):
    with open(file_path, "r") as file:
        data = json.load(file)

    # Convert each transaction into descriptive text
    texts = [
        f"Transaction ID {t['id']}: On {t['date']}, {t['customer']} purchased a {t['product']} for ₹{t['amount']}."
        for t in data
    ]

    return data, texts

# 2. Create Embeddings

def create_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    return model, embeddings


# 3. Retriever

def retrieve_transactions(query, model, embeddings, texts, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Pick top-k most similar transactions
    top_indices = similarities.argsort()[::-1][:top_k]
    retrieved = [texts[i] for i in top_indices]

    return retrieved

# 4. LLM-Based Generator (Real RAG)

def generate_answer(question, context_list):

    context = "\n".join(context_list)

    prompt = f"""
You are a strict RAG chatbot. 
You MUST answer the user's question using ONLY the transaction details provided below.
If the answer cannot be found in the context, say:
"Information not available in the provided transactions."

Context:
{context}

Question:
{question}

If needed, calculate totals, averages, or counts ONLY using this context.
Give a clear and useful answer.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message["content"]


# 5. Chatbot Loop

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

        print("\n Retrieved Context:")
        for r in retrieved:
            print("-", r)

        print("\n Bot Answer:")
        print(answer)
        print("\n-----\n")


# Run


if __name__ == "__main__":
    chatbot()
