import os
import shutil
import streamlit as st
from pathlib import Path

from rag.loader import load_and_chunk
from rag.chat import KnowledgeChatbot
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Constants
VECTORSTORE_DIR = "data/faiss_index"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Set up Streamlit UI
st.set_page_config(page_title="Knowledge Chatbot", layout="wide")
st.title("📚 AI-Powered Document Chatbot")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File uploader
st.sidebar.title("📄 Upload a Document")
uploaded_file = st.sidebar.file_uploader("Upload PDF, TXT, or MD file", type=["pdf", "txt", "md"])

# Process upload
if uploaded_file:
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.info(f"Processing `{uploaded_file.name}`...")
    chunks = load_and_chunk(file_path)
    st.sidebar.success(f"✅ Document loaded and split into {len(chunks)} chunks.")

    # Embeddings
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
    new_dim = len(embeddings.embed_query("test"))

    index_path = Path(VECTORSTORE_DIR)
    rebuild_index = False

    if index_path.exists():
        try:
            vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
            if vectorstore.index.d != new_dim:
                st.sidebar.warning("⚠️ Index dimension mismatch. Rebuilding...")
                shutil.rmtree(VECTORSTORE_DIR)
                rebuild_index = True
            else:
                vectorstore.add_documents(chunks)
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load existing index: {e}")
            shutil.rmtree(VECTORSTORE_DIR)
            rebuild_index = True

    if not index_path.exists() or rebuild_index:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTORSTORE_DIR)
    st.sidebar.success("📦 FAISS index updated.")

# Load chatbot
if Path(VECTORSTORE_DIR).exists():
    bot = KnowledgeChatbot(faiss_index_path=VECTORSTORE_DIR)
else:
    st.warning("Please upload a document to initialize the chatbot.")
    st.stop()

# Chat input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question about your document:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    with st.spinner("Thinking..."):
        response = bot.ask(user_input)
        st.session_state.chat_history.append(("🧑 You", user_input))
        st.session_state.chat_history.append(("🤖 Bot", response))

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    for role, message in reversed(st.session_state.chat_history):
        st.markdown(f"**{role}:** {message}")
