# 🎓 Edualze AI Knowledge Assistant

An AI-powered GenAI RAG (Retrieval-Augmented Generation) application that enables users to upload PDFs, add website URLs, and search Wikipedia topics to perform intelligent semantic question answering from custom knowledge sources.

Built using modern AI engineering concepts including:

* RAG Architecture
* Embeddings
* Vector Databases
* Semantic Search
* LLM Integration
* Prompt Engineering

---

# 🚀 Project Overview

Edualze AI Knowledge Assistant is a beginner-to-intermediate GenAI project designed to simulate how modern AI assistants retrieve and generate answers from external knowledge sources instead of relying only on pretrained model memory.

The application:

* Extracts content from PDFs, websites, and Wikipedia
* Splits text into semantic chunks
* Converts chunks into embeddings
* Stores vectors using FAISS
* Performs similarity search
* Retrieves relevant chunks
* Sends retrieved context to Gemini/OpenAI LLMs
* Generates contextual AI responses

This project demonstrates real-world implementation of:

* Retrieval-Augmented Generation (RAG)
* Vector similarity search
* Embedding-based semantic retrieval
* LLM orchestration using LangChain

---

# 🧠 Core AI Workflow

```text id="3zxaq7"
PDF / Website / Wikipedia
            ↓
      Text Extraction
            ↓
    Recursive Chunking
            ↓
        Embeddings
            ↓
      FAISS Vector DB
            ↓
      Similarity Search
            ↓
   Relevant Chunk Retrieval
            ↓
     Prompt Augmentation
            ↓
 Gemini / OpenAI LLM
            ↓
     Contextual Response
```

---

# ✨ Features

## 📄 Knowledge Source Ingestion

* Upload PDF study materials
* Add website/blog URLs
* Search Wikipedia topics dynamically

## 🔍 AI Semantic Search

* Embedding-based similarity retrieval
* Context-aware semantic understanding
* Top relevant chunk retrieval

## 🤖 AI-Powered Question Answering

* Ask questions from uploaded content
* Context-based LLM response generation
* Reduced hallucination using RAG

## 📝 AI Study Tools

* Automatic document summarization
* Quiz generation from study material
* Topic-wise AI assistance

## 💾 Vector Database Integration

* FAISS vector storage
* Persistent PKL vector indexing
* Efficient retrieval pipeline

## 🎨 Interactive UI

* Streamlit web interface
* Sidebar-based source management
* Real-time AI responses

---

# 🛠️ Tech Stack

| Category               | Technologies Used                                    |
| ---------------------- | ---------------------------------------------------- |
| Programming Language   | Python                                               |
| Frontend / UI          | Streamlit                                            |
| LLM Framework          | LangChain                                            |
| LLM APIs               | Gemini API / OpenAI API                              |
| Vector Database        | FAISS                                                |
| Embedding Models       | OpenAI Embeddings / HuggingFace Embeddings           |
| Document Loaders       | PyPDFLoader, WikipediaLoader, WebBaseLoader          |
| Text Processing        | RecursiveCharacterTextSplitter                       |
| Data Persistence       | Pickle (PKL)                                         |
| Environment Management | python-dotenv                                        |
| AI Concepts            | RAG, Embeddings, Semantic Search, Prompt Engineering |

---

# 📚 Key GenAI Concepts Implemented

## 🔹 Retrieval-Augmented Generation (RAG)

Combines retrieval systems with Large Language Models to generate answers grounded in external knowledge sources.

## 🔹 Embeddings

Converts text into numerical vectors representing semantic meaning.

## 🔹 Semantic Search

Searches based on meaning rather than exact keywords.

## 🔹 FAISS Vector Database

Stores high-dimensional embeddings and performs fast similarity search.

## 🔹 Chunking

Splits large documents into smaller semantic chunks for improved retrieval accuracy.

## 🔹 Prompt Engineering

Injects retrieved context into LLM prompts to generate accurate contextual responses.

---

# 📁 Project Structure

```text id="q8n4y7"
edualze-ai-knowledge-assistant/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env
├── study_materials/
├── vector_index.pkl
└── venv/
```

---

# ⚙️ Installation Guide

## 1️⃣ Clone Repository

```bash id="cxo3nt"
git clone https://github.com/ritimatangi/-EDUALZE-AI-KNOWLEDGE-ASSISTANT.git
cd -EDUALZE-AI-KNOWLEDGE-ASSISTANT
```

---

## 2️⃣ Create Virtual Environment

```bash id="40u6p7"
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash id="kmj8d6"
pip install -r requirements.txt
```

---

## 4️⃣ Configure API Key

Create a `.env` file:

```env id="bt7prm"
GOOGLE_API_KEY=your_api_key
```

OR

```env id="5jv7n7"
OPENAI_API_KEY=your_api_key
```

---

## 5️⃣ Run Application

```bash id="b8k9yt"
streamlit run app.py
```

---

# 🖥️ Application Usage

1. Upload PDF documents
2. Add website URLs
3. Search Wikipedia topics
4. Process knowledge sources
5. Ask questions
6. Generate summaries and quizzes

---

# 📄 Resume Description

### Edualze AI Knowledge Assistant — GenAI RAG Application

* Developed an AI-powered Retrieval-Augmented Generation (RAG) system using Python, LangChain, FAISS, Streamlit, and Gemini/OpenAI APIs
* Implemented semantic search pipeline using embeddings, vector similarity retrieval, and prompt-based contextual response generation
* Built multi-source knowledge ingestion from PDFs, websites, and Wikipedia
* Integrated AI-powered summarization, quiz generation, and semantic question answering features
* Designed vector persistence workflow using Pickle (PKL) for efficient retrieval operations

---

# 🔮 Future Improvements

* Conversation Memory
* Voice-Based Interaction
* Multi-User Authentication
* Cloud Deployment
* Hybrid Search
* MongoDB Integration
* Advanced Reranking
* Agentic AI Workflows

---

# 👨‍💻 Author

## Riti Matangi

GitHub: https://github.com/ritimatangi

