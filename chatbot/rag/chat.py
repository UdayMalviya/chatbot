import os
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from rag.retriver import DocumentRetriever

class KnowledgeChatbot:
    def __init__(self, faiss_index_path: str, model: str = None):
        self.retriever = DocumentRetriever(faiss_index_path)
        self.llm = ChatOllama(
            model=model or os.getenv("OLLAMA_MODEL", "mistral"),
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434")
        )

    def ask(self, query: str, k: int = 4) -> str:
        query = query.strip()
        if not query:
            return "❗ Please enter a valid question."

        docs = self.retriever.retrieve(query, k)
        if not docs:
            return "🤖 I couldn't find anything relevant in the uploaded file."

        context = "\n\n".join(doc.page_content for doc in docs)
        system_prompt = (
            "You are a helpful assistant. Use only the following context to answer. "
            "If the answer is not in the context, say 'I don't know'.\n\n"
            f"Context:\n{context}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return f"❌ Error generating response: {e}"
