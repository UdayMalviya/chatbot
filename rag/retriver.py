import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

class DocumentRetriever:
    def __init__(self, faiss_index_path: str):
        model = os.getenv("OLLAMA_MODEL", "mistral")
        base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.embeddings = OllamaEmbeddings(model=model, base_url=base_url)
        self.db = FAISS.load_local(faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        return self.db.similarity_search(query, k=k)
