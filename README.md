<h1>RAG Transaction Chatbot</h1>

This project is a simple RAG (Retrieval Augmented Generation) based chatbot that can answer questions about customer transactions.
I built it using Python, SentenceTransformers, and Streamlit.

The chatbot can run in two ways:

main.py → terminal/console version

app.py → Streamlit web UI version

<h2>What This Project Does</h2>

* Loads transaction data from a JSON file

* Converts each transaction into a readable sentence

* Creates embeddings using a pretrained model

* Searches for the most relevant transactions using cosine similarity

* Filters answers based on the customer (Amit, Riya, Karan)

* Shows the result either in the terminal or a Streamlit web page

<h3>Basically, you can ask things like:</h3>

“Tell me Amit purchase history”

“Show Riya’s transactions”

“What did Karan buy?”

And the chatbot will return the matching transactions.

Dataset Used (transactions.json)

I used a small sample dataset:

<h4>[
    {"id": 1, "customer": "Amit", "product": "Laptop", "amount": 55000, "date": "2024-01-12"},
    {"id": 2, "customer": "Amit", "product": "Mouse", "amount": 700, "date": "2024-02-15"},
    {"id": 3, "customer": "Riya", "product": "Mobile", "amount": 30000, "date": "2024-01-05"},
    {"id": 4, "customer": "Riya", "product": "Earbuds", "amount": 1500, "date": "2024-02-20"},
    {"id": 5, "customer": "Karan", "product": "Keyboard", "amount": 1200, "date": "2024-03-01"}
]</h4>

Requirements

<h3>To run the project, I installed these:</h3>

pip install sentence-transformers
pip install numpy
pip install scikit-learn
pip install streamlit

How the Project Works (Short Explanation)

Load the JSON dataset

Create meaningful text for each transaction

Generate embeddings for all transactions

Take the user’s question

Convert the question into an embedding

Compare the question embedding with transaction embeddings

Retrieve the most similar ones

Filter by customer name if the question mentions it

Show the final answer

This is basically the RAG workflow but in a simple way.

Running the Terminal Version
python main.py


The terminal will ask for your question and show the results there.

Running the Streamlit App
streamlit run app.py
