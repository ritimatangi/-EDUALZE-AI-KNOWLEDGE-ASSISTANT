# Edualze AI Knowledge Assistant

A beginner-friendly GenAI RAG (Retrieval-Augmented Generation) application that allows users to upload PDFs, add website URLs, and search Wikipedia topics to ask intelligent questions from their own knowledge sources.

---

## Project Overview

Edualze AI Knowledge Assistant is an AI-powered study assistant built using:

* LangChain
* FAISS Vector Database
* Streamlit
* Gemini/OpenAI APIs
* RAG Architecture

The application extracts content from multiple sources, converts it into embeddings, stores it in a FAISS vector database, and retrieves relevant chunks to generate context-aware AI responses.

---

## Features

* PDF Upload & Processing
* Website URL Content Extraction
* Wikipedia Topic Search
* AI-Powered Question Answering
* Semantic Search using Embeddings
* FAISS Vector Database Integration
* Document Summarization
* Quiz Generation from Study Material
* Source Attribution for Answers
* Persistent Vector Storage using PKL
* Streamlit-Based Interactive UI

---

## RAG Workflow

```text
PDF / Website / Wikipedia
            ↓
      Text Extraction
            ↓
          Chunking
            ↓
         Embeddings
            ↓
      FAISS Vector DB
            ↓
      Similarity Search
            ↓
     Relevant Chunk Retrieval
            ↓
     LLM Context Injection
            ↓
      AI Generated Answer
```

---

## Tech Stack

| Technology      | Purpose                   |
| --------------- | ------------------------- |
| Python          | Backend Development       |
| Streamlit       | Interactive Web UI        |
| LangChain       | LLM Orchestration         |
| FAISS           | Vector Database           |
| Gemini / OpenAI | LLM Response Generation   |
| Pickle (PKL)    | Persistent Vector Storage |

---

## Key Concepts Used

### RAG (Retrieval-Augmented Generation)

Improves AI responses by retrieving relevant information from uploaded documents before generating answers.

### Embeddings

Converts text into numerical vectors for semantic similarity search.

### FAISS

Efficient vector database used for storing and retrieving embeddings.

### Chunking

Large documents are split into smaller chunks for better retrieval accuracy.

### Semantic Search

Searches by meaning instead of exact keywords.

---

## Project Structure

```text
edualze-ai-knowledge-assistant/
│
├── app.py
├── README.md
├── requirements.txt
├── .env
├── .gitignore
├── study_materials/
└── vector_index.pkl
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/ritimatangi/-EDUALZE-AI-KNOWLEDGE-ASSISTANT.git
cd -EDUALZE-AI-KNOWLEDGE-ASSISTANT
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API Key

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key
```

or

```env
OPENAI_API_KEY=your_api_key
```

### 5. Run Application

```bash
streamlit run app.py
```

---

## Usage

1. Upload PDF documents
2. Add website URLs or Wikipedia topics
3. Process knowledge sources
4. Ask questions from uploaded content
5. Generate summaries and quizzes

---

## Interview Explanation

Built a RAG-based AI Knowledge Assistant using LangChain, FAISS, Streamlit, and Gemini/OpenAI APIs. The application performs text extraction, chunking, embedding generation, semantic similarity retrieval, and context-aware response generation using vector databases and LLMs.

---

## Resume Description

**Edualze AI Knowledge Assistant**
Developed a Retrieval-Augmented Generation (RAG) based AI learning assistant using Python, LangChain, FAISS, and Gemini/OpenAI APIs. Implemented semantic search, vector embeddings, document retrieval, and AI-powered question answering from PDFs, websites, and Wikipedia sources.

---

## Future Improvements

* Conversation Memory
* Voice-Based Question Answering
* Cloud Deployment
* Multi-User Support
* Hybrid Search
* MongoDB Integration

---

## Author

Riti Matangi

GitHub: https://github.com/ritimatangi
