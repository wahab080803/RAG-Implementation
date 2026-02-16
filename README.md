# RAG-Implementation
Successfully completed the development and evaluation of the Full RAG System as assigned. The project is structured into a modular pipeline to ensure transparency, hardware efficiency, and factual accuracy.

# chunking.py
Purpose: This script performs the initial Chunking Strategy Comparison by loading the data.pdf and testing two different methods: Fixed-Size splitting versus Recursive splitting.
Key Function: It identifies that the Recursive strategy is superior for technical documents because it preserves logical boundaries like paragraphs and sentences rather than cutting text at random points.

# embeddings_test.py
Purpose: This file executes the Embedding Comparison requirement by converting your text chunks into mathematical vectors using two different AI models: all-MiniLM-L6-v2 and bge-small-en-v1.5.
Key Function: It creates and persists (saves) two separate vector databases to your disk (./db_minilm and ./db_bge) so the search engine can retrieve information later without re-processing the PDF.

#  retrieval_eval.py
Purpose: This script handles the Retrieval Evaluation by measuring the "Hit Rate" of your search engine—specifically checking if the system finds the correct page for a set of "Golden Questions".
Key Function: It uses the existing ChromaDB databases to calculate a numerical accuracy score, proving which embedding model (MiniLM or BGE) is more reliable at finding facts.

#  final_rag_eval.py
Purpose: This file generates the Response Evaluation Metrics by scoring the AI's final answer against the original context.
Key Function: It calculates Faithfulness (checking if the AI lied or stayed grounded in the PDF) and Context Precision to provide a final "Report Card" for your RAG system.

#  chat_with_pdf.py
Purpose: This is the Interactive Interface that allows a user to ask questions in real-time through the terminal.
Key Function: It performs a live search on your winning database (MiniLM) and prints the relevant answer snippets along with the exact Page Numbers where the information was found.
