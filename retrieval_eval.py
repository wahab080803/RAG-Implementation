import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Define the Embedding Models again (Same as before)
model_a_name = "sentence-transformers/all-MiniLM-L6-v2"
model_b_name = "BAAI/bge-small-en-v1.5"

embeddings_a = HuggingFaceEmbeddings(model_name=model_a_name)
embeddings_b = HuggingFaceEmbeddings(model_name=model_b_name)

# 2. LOAD the existing databases from your disk
# Use the same persist_directory paths from your previous script
db_a = Chroma(persist_directory="./db_minilm", embedding_function=embeddings_a)
db_b = Chroma(persist_directory="./db_bge", embedding_function=embeddings_b)

# 3. Define your Evaluation Set (Golden Questions)
# Since your PDF is about Hadoop (19 pages), pick questions from known pages.
eval_set = [
    {"question": "Who is Dr. Amr Awadallah?", "expected_page": 1},
    {"question": "When should Hadoop be used versus a Data Warehouse?", "expected_page": 1},
    {"question": "What role does Teradata play in this enterprise system?", "expected_page": 1}
]

# 4. Define the Hit Rate function
def calculate_hit_rate(vector_db, eval_set, k=3):
    hits = 0
    for item in eval_set:
        # Search for the top k chunks
        results = vector_db.similarity_search(item["question"], k=k)
        # Extract page numbers from metadata (pypdf starts at 0, so we add 1)
        retrieved_pages = [res.metadata.get('page', -1) + 1 for res in results]
        
        if item["expected_page"] in retrieved_pages:
            hits += 1
            print(f"✓ Match for: '{item['question']}' on Page {item['expected_page']}")
        else:
            print(f"✗ Miss for: '{item['question']}' (Found pages: {retrieved_pages})")
            
    return (hits / len(eval_set)) * 100

# 5. Run the Evaluation
print("\n--- Evaluating Model A (MiniLM) ---")
hr_a = calculate_hit_rate(db_a, eval_set)

print("\n--- Evaluating Model B (BGE) ---")
hr_b = calculate_hit_rate(db_b, eval_set)

print(f"\nFinal Results:")
print(f"MiniLM Hit Rate@3: {hr_a}%")
print(f"BGE Hit Rate@3: {hr_b}%")