import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. LOAD & CHUNK (Step 1 from chunking.py)
print("Loading and chunking PDF...")
loader = PyPDFLoader("data.pdf")
docs = loader.load()

# Recursive strategy (The 100 chunks success)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
recursive_chunks = splitter.split_documents(docs)

# 2. DEFINE EMBEDDING MODELS (Requirement: Embedding Comparison)
model_a_name = "sentence-transformers/all-MiniLM-L6-v2"
model_b_name = "BAAI/bge-small-en-v1.5"

print(f"Initializing Model A: {model_a_name}")
embeddings_a = HuggingFaceEmbeddings(model_name=model_a_name)

print(f"Initializing Model B: {model_b_name}")
embeddings_b = HuggingFaceEmbeddings(model_name=model_b_name)

# 3. CREATE VECTOR DATABASES (Requirement: Retrieval Evaluation)
print("\nCreating Vector DB for Model A...")
db_a = Chroma.from_documents(
    documents=recursive_chunks, 
    embedding=embeddings_a,
    persist_directory="./db_minilm"
)

print("Creating Vector DB for Model B...")
db_b = Chroma.from_documents(
    documents=recursive_chunks, 
    embedding=embeddings_b,
    persist_directory="./db_bge"
)

# 4. TEST RETRIEVAL COMPARISON
query = "What does Amr Awadallah say about Hadoop?"
print(f"\n--- Query: {query} ---")

results_a = db_a.similarity_search(query, k=1)
results_b = db_b.similarity_search(query, k=1)

print(f"\n--- Model A Result ({model_a_name}) ---")
print(results_a[0].page_content[:300] + "...")

print(f"\n--- Model B Result ({model_b_name}) ---")
print(results_b[0].page_content[:300] + "...")