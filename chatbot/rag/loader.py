from pathlib import Path
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader
)

EXTENSION_TO_LOADER = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
}

def load_file(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()
    loader_class = EXTENSION_TO_LOADER.get(ext, UnstructuredFileLoader)
    loader = loader_class(file_path)
    return loader.load()

def chunk_documents(documents: List[Document], chunk_size=500, chunk_overlap=100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def load_and_chunk(file_path: str) -> List[Document]:
    docs = load_file(file_path)
    return chunk_documents(docs)
