# rag/embedder.py
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
# from loader import load_documents, chunk_documents
from rag.loader import load_file, chunk_documents
import os

def embed_and_store(doc_dir: str, faiss_index_path: str):
    # Load and chunk docs
    raw_docs = load_file(doc_dir)
    chunks = chunk_documents(raw_docs)

    # Create embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Build FAISS index
    db = FAISS.from_documents(chunks, embeddings)

    # Save index
    db.save_local(faiss_index_path)
    print(f"Saved FAISS index to: {faiss_index_path}")
