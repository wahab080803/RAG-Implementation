import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

# 1. Load the PDF
loader = PyPDFLoader("data.pdf")
docs = loader.load()

# 2. Strategy A: Fixed-Size Chunking (Basic)
# Breaks text every 500 characters, no matter what.
fixed_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
fixed_chunks = fixed_splitter.split_documents(docs)

# 3. Strategy B: Recursive Character Chunking (Recommended)
# Breaks at paragraphs, then sentences, then words. Better for context.
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
recursive_chunks = recursive_splitter.split_documents(docs)

# 4. Compare the results
print(f"Original Pages: {len(docs)}")
print(f"Fixed Chunks: {len(fixed_chunks)}")
print(f"Recursive Chunks: {len(recursive_chunks)}")

# Show a sample from the Recursive Strategy
print("\n--- Sample Chunk (Recursive) ---")
print(recursive_chunks[0].page_content)