
# 🤖 RAG-Based Technical Document Assistant

A secure, local Retrieval-Augmented Generation (RAG) app that helps you analyze and extract insights from structured technical documents (like Use Case Specifications).

---

## 🚀 Features

- 📤 Upload technical PDFs (e.g., Use Case Specs)
- 📚 Embed & store vectors with ChromaDB
- 🧠 Query using a local LLM (via Ollama)
- 🏆 Re-rank results using sentence-transformer cross-encoders
- 📝 Extract BRs, AFs, XFs, and system message logic from main flows
- 🔒 Fully local/offline setup – no data ever leaves your system

---

## 📦 Prerequisites

Before running the project, make sure the following are installed:

### 🛠️ System Requirements

- Python 3.11
- [Ollama](https://ollama.com/) (for running local LLMs)

> 🧪 Install Ollama:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

> After installation, start Ollama manually:
```bash
ollama serve
```

Ollama will run a local server on `http://localhost:11434`.

---

## 🤖 Ollama Model Setup

This project requires the following models to be pulled via Ollama:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text:latest
```

These are used for:
- `llama3.2:3b` → main reasoning LLM
- `nomic-embed-text:latest` → embedding function for ChromaDB

Make sure `ollama serve` is running before starting the app.

---

## ⚙️ Setup

All project tasks are defined in the `Makefile`.

### 🔹 1. Create Virtual Environment & Install Dependencies

```bash
make setup
```

This will:
- Create a `.venv` virtual environment
- Install all Python dependencies inside it

> Activate your environment:
```bash
source .venv/bin/activate
```

---

## 🚀 Run the App

```bash
make run
```

Then open your browser and visit:  
📍 `http://localhost:8501`

---

## 🧾 Use Case Flow

1. **Upload**: Drop in a use case PDF
2. **Process**: Text is chunked & embedded to ChromaDB
3. **Query**: Ask questions using natural language
4. **Re-rank**: Top results are ranked using CrossEncoder
5. **LLM**: Final answer generated using selected local LLM

---

## 📋 Custom Prompt Logic

The assistant will:
- Extract document name from "Use Case Specification" header (e.g., `BNYM_ABC_001`)
- Identify referenced/extension documents in the "Brief Description"
- Answer using only the provided context — no external knowledge

You can modify the `system_prompt` in `app.py` to change behavior.

---

## 📁 Project Structure

```bash
ProjectAI/
├── app.py                 # Streamlit app with RAG pipeline
├── Makefile               # All project tasks
├── requirements.txt       # Python dependencies
├── demo-rag-chroma/       # ChromaDB persistent store
└── README.md              # This file
```

---

## 🛡️ Security

✅ 100% local  
✅ Offline document parsing  
✅ Secure embedding and querying  

---

## 📬 Contribute / Extend

This app is modular. You can easily extend it to:
- Support DOCX or HTML documents
- Compare across document layers (product → sub-product → bank)
- Output auto-formatted documentation

PRs welcome!

---

### Built for internal document reasoning, legacy analysis, and secure AI workflows.
