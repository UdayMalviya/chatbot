# 🧠 AI-Powered Document Chatbot

A fully offline, local Retrieval-Augmented Generation (RAG) chatbot that can answer questions based on any uploaded document (PDF, TXT, MD) using [Ollama](https://ollama.com/) + [FAISS](https://github.com/facebookresearch/faiss) + [LangChain](https://www.langchain.com/) + [Streamlit](https://streamlit.io/).

---

## 🚀 Features

- 📄 Upload and read PDF, TXT, or Markdown files
- 📚 Chunk and embed documents using `OllamaEmbeddings`
- 🧠 Store and retrieve from FAISS vector store
- 💬 Ask questions and receive context-aware answers
- 🔌 Runs fully offline using local Ollama models (like `mistral`, `phi`, `gemma`)
- 🖥️ Clean Streamlit-based UI

---

## 📁 Project Structure

```

chatbot/
├── app.py                  # Streamlit app
├── rag/
│   ├── chat.py             # Chat logic (LLM + prompt)
│   ├── retriver.py         # FAISS + OllamaEmbeddings
│   └── loader.py           # File loader + chunking
├── data/faiss\_index/       # FAISS index folder
├── temp/                   # Uploaded raw files
└── requirements.txt        # Dependencies

````

---

## ⚙️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/chatbot.git
cd chatbot
````

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama and pull a model

Install Ollama from [https://ollama.com](https://ollama.com) and run:

```bash
ollama pull mistral
```

Or use a faster model like:

```bash
ollama pull phi
```

---

## 🔧 Configuration

Set environment variables (or use `.env`):

```bash
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=mistral
```

These are optional — defaults will be used if not set.

---

## 🧠 Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Dependencies

Key packages:

* `streamlit`
* `langchain`
* `langchain-community`
* `langchain-ollama`
* `faiss-cpu`
* `sentence-transformers`
* `unstructured`
* `PyMuPDF`

Install all with:

```bash
pip install -r requirements.txt
```

---

## 🛡️ License

MIT License. Use responsibly.

---

## 🙌 Credits

Built by Uday using:

* LangChain
* Ollama
* FAISS
* Streamlit

