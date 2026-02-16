import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. LOAD YOUR WINNING MODEL (MiniLM)
model_a_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_a = HuggingFaceEmbeddings(model_name=model_a_name)

# 2. CONNECT TO YOUR EXISTING DATABASE
# This uses the database you already created in embeddings_test.py
db_a = Chroma(persist_directory="./db_minilm", embedding_function=embeddings_a)

def start_chat():
    print("\n" + "="*50)
    print("PDF SEARCH ENGINE READY (Based on data.pdf)")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50)

    while True:
        user_query = input("\nYour Question: ")
        
        if user_query.lower() in ['exit', 'quit']:
            break

        # 3. PERFORM SEARCH
        # We retrieve the top 2 most relevant chunks
        results = db_a.similarity_search(user_query, k=2)

        if not results:
            print("No relevant information found in the document.")
            continue

        print("\n--- Answer/Found Context ---")
        for i, doc in enumerate(results):
            # Extract the page number from metadata
            # We add +1 because pypdf is 0-indexed
            page_num = doc.metadata.get('page', -1) + 1
            
            print(f"[{i+1}] Found on Page {page_num}:")
            print(f"    {doc.page_content[:400]}...") # Showing snippet of the answer
            print("-" * 20)

if __name__ == "__main__":
    start_chat()