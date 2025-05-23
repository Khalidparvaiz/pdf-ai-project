# 📚 PDF AI Agent

An intelligent AI-powered app that lets you **ask questions directly from your PDF documents** using advanced Retrieval-Augmented Generation (RAG) techniques. Built with 🔥 LangChain, 💬 Ollama (Gemma), and ⚡ Streamlit.

> “Upload PDFs. Ask Anything. Get Instant Answers.”

---

## 🌟 Features

- Upload multiple PDF files at once
- Extract and chunk content using PDFPlumber and LangChain
- Generate embeddings with Ollama's Gemma model
- Retrieve answers using MultiQueryRetriever and Chroma vector store
- Streamlit frontend for easy interaction

---

## 🛠️ Installation

1. Clone the repo  
2. Install required Python packages  
3. Ensure Ollama is installed and running (Gemma model)  
4. Launch the Streamlit app

```bash
# Step 1: Clone the repo
git clone https://github.com/yourusername/pdf-ai-project.git
cd pdf-ai-project

# Step 2: Install required Python packages
pip install -r requirements.txt

# Step 3: Ensure Ollama is installed and running locally (Gemma model)
ollama run gemma:2b

# Step 4: Launch the Streamlit app
streamlit run app.py
