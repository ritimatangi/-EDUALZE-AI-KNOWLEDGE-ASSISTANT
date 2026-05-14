# 🎓 Edualze AI Knowledge Assistant

> A beginner-friendly AI-powered study assistant using RAG (Retrieval-Augmented Generation). Upload PDFs, paste website URLs, or search Wikipedia — then ask questions and get intelligent answers from YOUR content!

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-purple?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-orange?style=flat-square&logo=openai)

---

## 📋 Table of Contents

1. [What is This Project?](#-what-is-this-project)
2. [What is RAG?](#-what-is-rag)
3. [Architecture Diagram](#-architecture-diagram)
4. [Key Concepts Explained](#-key-concepts-explained)
5. [Features](#-features)
6. [Project Structure](#-project-structure)
7. [Setup Guide](#-setup-guide)
8. [How to Get an OpenAI API Key](#-how-to-get-an-openai-api-key)
9. [How to Run](#-how-to-run)
10. [Common Errors and Fixes](#-common-errors-and-fixes)
11. [Interview Explanation](#-interview-explanation)
12. [Resume Description](#-resume-description)

---

## 🤔 What is This Project?

**Edualze AI Knowledge Assistant** is a **RAG application** — a smart study buddy that:

1. **Reads** your study materials (PDFs, websites, Wikipedia articles)
2. **Understands** the content by converting it into searchable numbers (embeddings)
3. **Remembers** everything in a vector database (FAISS), saved as a PKL file
4. **Answers** your questions using ONLY YOUR materials (not making things up!)

---

## 🧠 What is RAG?

**RAG = Retrieval-Augmented Generation**

```
Without RAG:  Question → AI Brain → Answer (might hallucinate ❌)
With RAG:     Question → Search YOUR docs → Found text → AI reads it → Answer (accurate ✅)
```

**Analogy:** Two students taking an exam:
- **Student A (no RAG):** Answers from memory → might forget or make up answers
- **Student B (RAG):** Opens the textbook first, finds relevant paragraphs, THEN answers → accurate!

---

## 🏗️ Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    USER INTERFACE                         │
│                  (Streamlit Web App)                      │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────┐     │
│  │ Upload   │  │ Paste URL    │  │ Search         │     │
│  │ PDFs     │  │              │  │ Wikipedia      │     │
│  └────┬─────┘  └──────┬───────┘  └───────┬────────┘     │
└───────┼───────────────┼──────────────────┼──────────────┘
        │               │                  │
        ▼               ▼                  ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 1: TEXT EXTRACTION                                  │
│  PyPDFLoader  /  WebBaseLoader  /  WikipediaLoader        │
└─────────────────────┬────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 2: CHUNKING — RecursiveCharacterTextSplitter        │
│  (1000 chars per chunk, 200 char overlap)                 │
└─────────────────────┬────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 3: EMBEDDINGS — OpenAIEmbeddings                    │
│  (text → 1536-dimensional number vectors)                 │
└─────────────────────┬────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 4: VECTOR STORAGE — FAISS + vector_index.pkl        │
└─────────────────────┬────────────────────────────────────┘
                      │  User asks a question
                      ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 5: SIMILARITY SEARCH — Top-4 closest chunks         │
└─────────────────────┬────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 6: LLM — GPT-3.5-turbo answers from context        │
└─────────────────────┬────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────┐
│  RESPONSE: Answer + Source Chunks + Source Labels          │
│  (📄 PDF  /  🌐 Website  /  📚 Wikipedia)                │
└──────────────────────────────────────────────────────────┘
```

---

## 📚 Key Concepts Explained

### ✂️ What is Chunking?
Breaking long text into small pieces (~1000 characters). **Why?** AI has a token limit, and smaller chunks give more precise search results.

### 🔢 What are Embeddings?
Numbers that represent the MEANING of text. "cat" and "kitten" get similar numbers because they mean similar things. This enables **semantic search** (search by meaning, not keywords).

### 💾 What is FAISS?
Facebook AI Similarity Search — a database that stores vectors and finds the most similar ones in milliseconds.

### 📦 Why PKL (Pickle)?
Saves your FAISS index to a file so you don't need to recompute embeddings every time. Just 2 lines: `pickle.dump()` to save, `pickle.load()` to restore.

### 🔍 What is Similarity Search?
Converting your question to a vector, then finding stored chunks whose vectors are closest (most similar in meaning).

### 🛡️ How RAG Reduces Hallucination
The prompt says "Answer ONLY from the provided context." The AI can't make up facts because it's restricted to your actual documents!

---

## ✨ Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | 📄 PDF Upload | Upload one or multiple PDF study materials |
| 2 | 🌐 URL Ingestion | Paste any blog/article URL to add content |
| 3 | 📚 Wikipedia Search | Search any topic and fetch Wikipedia articles |
| 4 | ✂️ Smart Chunking | Breaks text into 1000-char searchable pieces |
| 5 | 🔢 Embeddings | Converts text into 1536-dim meaning vectors |
| 6 | 💾 FAISS Store | Stores vectors for fast similarity search |
| 7 | 📦 PKL Persistence | Saves vectors to vector_index.pkl file |
| 8 | 💬 QA Chatbot | Ask questions, get AI answers from YOUR docs |
| 9 | 📖 Source Attribution | Shows which chunks were used (PDF/Web/Wiki) |
| 10 | 📝 PDF Summary | Generate bullet-point summaries |
| 11 | ❓ Quiz Generator | Create MCQ questions from study materials |
| 12 | 🗑️ Clear Chat | Reset conversation |
| 13 | 🔄 Reset KB | Clear entire knowledge base |
| 14 | 🎨 Modern UI | Gradient header, styled sidebar, source badges |

---

## 📁 Project Structure

```
edualze-ai-assistant/
├── app.py                  ← The ENTIRE application (single file)
├── requirements.txt        ← Python dependencies
├── .env                    ← Your OpenAI API key
├── .gitignore              ← Files to exclude from Git
├── README.md               ← This file!
├── study_materials/        ← Uploaded PDFs (auto-created)
└── vector_index.pkl        ← Saved FAISS vectors (auto-created)
```

---

## 🚀 Setup Guide

### Step 1: Install Python
```bash
python3 --version   # Should show Python 3.9+
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up API Key
```bash
# Open .env and replace "your-api-key-here" with your actual key
# OR just enter it in the app's sidebar
```

### Step 5: Run!
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` 🎉

---

## 🔑 How to Get an OpenAI API Key

1. Go to https://platform.openai.com/signup
2. Create an account
3. Go to https://platform.openai.com/api-keys
4. Click **"Create new secret key"**
5. Copy the key (you won't see it again!)
6. Paste in `.env` or the app sidebar

**Cost:** ~$0.10 per study session (very affordable!)

---

## 🏃 How to Run

```bash
cd "edualze gen ai"
source venv/bin/activate
streamlit run app.py
```

**Usage:**
1. Enter your OpenAI API key in the sidebar
2. Upload a PDF / paste a URL / search Wikipedia
3. Click the process button
4. Start asking questions!

---

## ❌ Common Errors and Fixes

| Error | Fix |
|-------|-----|
| `openai.AuthenticationError` | Check your API key |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `No text extracted from PDF` | Use text-based PDFs (not scanned images) |
| `Rate limit exceeded` | Wait a minute and retry |
| `URL loading failed` | Try a different publicly accessible URL |
| `Wikipedia topic not found` | Try a more specific topic name |
| `streamlit: command not found` | Activate venv first: `source venv/bin/activate` |

---

## 🎤 Interview Explanation

> "I built Edualze AI Knowledge Assistant — a RAG-based study tool. Users upload PDFs, paste URLs, or search Wikipedia. Text is extracted with LangChain loaders, chunked with RecursiveCharacterTextSplitter (1000 chars, 200 overlap), and converted to 1536-dim embeddings via OpenAI. Vectors are stored in FAISS and persisted as a PKL file. When a user asks a question, it's embedded, FAISS finds the top-4 most similar chunks, and those chunks + the question go to GPT-3.5-turbo with a prompt that restricts answers to the provided context — reducing hallucination. The app also generates summaries and quizzes. Tech stack: Python, Streamlit, LangChain, FAISS, OpenAI, Pickle."

---

## 📄 Resume Description

**Edualze AI Knowledge Assistant** — *RAG-based Study Tool*
- Built a Retrieval-Augmented Generation (RAG) application using Python, LangChain, FAISS, and OpenAI API
- Implemented multi-source knowledge ingestion from PDFs, websites, and Wikipedia
- Designed semantic search pipeline: text extraction → chunking → embeddings → FAISS → similarity retrieval → LLM answers
- Persisted vector store using Pickle (PKL) for offline access
- Added document summarization, quiz generation, and source attribution features
- **Tech Stack:** Python, Streamlit, LangChain, FAISS, OpenAI API, Pickle

---

<div align="center">
  <strong>Built with ❤️ for learners everywhere</strong><br>
  <em>Edualze AI Knowledge Assistant</em>
</div>
