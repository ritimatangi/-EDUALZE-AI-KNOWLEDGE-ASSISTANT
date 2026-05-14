# ============================================================
# 🎓 EDUALZE AI KNOWLEDGE ASSISTANT
# ============================================================
# A beginner-friendly RAG (Retrieval-Augmented Generation) app.
#
# WHAT THIS APP DOES:
#   1. User uploads PDFs, pastes URLs, or searches Wikipedia
#   2. Text is extracted and split into chunks
#   3. Chunks are converted to embeddings (number vectors)
#   4. Vectors are stored in FAISS and saved as a PKL file
#   5. User asks a question
#   6. FAISS finds the most similar chunks (retrieval)
#   7. Chunks + question are sent to Google Gemini (generation)
#   8. AI answers based ONLY on the provided content
#
# HOW TO RUN:
#   streamlit run app.py
# ============================================================


# ============================================================
# STEP 0: IMPORT ALL REQUIRED LIBRARIES
# ============================================================
# Each library has a specific job in our RAG pipeline.

import os                          # For file/folder operations
import pickle                     # For saving/loading FAISS index as PKL
import tempfile                   # For temporarily saving uploaded PDFs
import time                       # For retry delays when rate limited

import streamlit as st            # Web UI framework (creates the app interface)
from dotenv import load_dotenv    # Loads API key from .env file

# --- LangChain: The framework that connects all GenAI pieces ---
from langchain_text_splitters import RecursiveCharacterTextSplitter       # Chunking
from langchain_google_genai import GoogleGenerativeAIEmbeddings          # Embeddings (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI                # LLM (Gemini)
from langchain_community.vectorstores import FAISS                       # Vector DB
from langchain_community.document_loaders import PyPDFLoader        # PDF loader
from langchain_community.document_loaders import WebBaseLoader      # URL loader
from langchain_community.document_loaders import WikipediaLoader    # Wiki loader
from langchain_core.messages import HumanMessage, SystemMessage        # Chat messages

# ============================================================
# LOAD API KEY FROM .env FILE
# ============================================================
# HOW .env WORKS:
#   1. python-dotenv reads the .env file in your project folder
#   2. It loads all KEY=VALUE pairs as environment variables
#   3. os.getenv("KEY") reads the value
#   4. This keeps your API key SECRET — never hardcode it!
#
# Your .env file should look like:
#   GOOGLE_API_KEY=AIza...
# ============================================================
load_dotenv()

# Read the API key from environment variable
# os.getenv() returns None if the key is not found
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ============================================================
# STEP 1: PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Edualze AI Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STEP 2: CUSTOM CSS FOR MODERN STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    /* Gradient header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
    }
    .main-header h1 { color: white; font-size: 2rem; font-weight: 700; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.85); font-size: 1rem; margin: 0.3rem 0 0 0; }

    /* Source badges (PDF / Website / Wikipedia) */
    .source-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin: 2px 4px;
    }
    .source-pdf { background: rgba(234,88,12,0.15); color: #ea580c; border: 1px solid rgba(234,88,12,0.3); }
    .source-web { background: rgba(37,99,235,0.15); color: #2563eb; border: 1px solid rgba(37,99,235,0.3); }
    .source-wiki { background: rgba(22,163,74,0.15); color: #16a34a; border: 1px solid rgba(22,163,74,0.3); }

    /* Knowledge source cards */
    .knowledge-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 0.75rem 1rem; border-radius: 10px;
        border-left: 4px solid #667eea; margin: 0.5rem 0; font-size: 0.85rem;
    }

    /* Chunk display boxes */
    .chunk-box {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
        font-size: 0.85rem; line-height: 1.6;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%); }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown label { color: #e0e7ff !important; }

    /* Status indicators */
    .status-ready { color: #22c55e; font-weight: 600; }
    .status-empty { color: #f59e0b; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# STEP 3: SESSION STATE INITIALIZATION
# ============================================================
# Streamlit re-runs the ENTIRE script on every interaction.
# Session state keeps our data alive between reruns.

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # Stores chat messages

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None      # FAISS vector database

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []         # All text chunks (for summary/quiz)

if "knowledge_sources" not in st.session_state:
    st.session_state.knowledge_sources = []  # Track added sources


# ============================================================
# STEP 4: PROMPT TEMPLATES
# ============================================================
# These prompts tell the AI HOW to answer. This is "prompt engineering".
# Notice: "Use ONLY the provided context" — this prevents hallucination!

QA_PROMPT = """You are a helpful and friendly study assistant called Edualze AI.

Your job is to answer the student's question using ONLY the provided context.

RULES:
1. Use ONLY the information from the context below to answer.
2. If the context doesn't contain enough information, say:
   "I don't have enough information in the uploaded documents to answer this question. Try uploading more relevant materials!"
3. Be clear and beginner-friendly in your explanations.
4. Use bullet points or numbered lists when helpful.
5. If relevant, mention which part of the context your answer comes from.

CONTEXT (from uploaded documents):
{context}

PREVIOUS CONVERSATION:
{chat_history}

STUDENT'S QUESTION:
{question}

YOUR ANSWER:"""

SUMMARY_PROMPT = """Summarize the following study material in a clear, beginner-friendly way.

RULES:
1. Use bullet points for key concepts.
2. Keep the language simple and easy to understand.
3. Highlight the most important topics.
4. Organize the summary into logical sections if possible.
5. Keep the summary concise but comprehensive.

STUDY MATERIAL:
{text}

SUMMARY:"""

QUIZ_PROMPT = """Generate 5 multiple-choice quiz questions from the following study material.

RULES:
1. Each question should test understanding, not just memorization.
2. Provide 4 options (A, B, C, D) for each question.
3. Clearly mark the correct answer.
4. Add a brief explanation for why the correct answer is right.
5. Make questions progressively harder (easy → medium → hard).

FORMAT EACH QUESTION LIKE THIS:

**Question 1:** [Question text]
- A) [Option A]
- B) [Option B]
- C) [Option C]
- D) [Option D]

✅ **Correct Answer:** [Letter]) [Answer text]
💡 **Explanation:** [Brief explanation]

---

STUDY MATERIAL:
{text}

QUIZ QUESTIONS:"""


# ============================================================
# STEP 5: HELPER FUNCTIONS
# ============================================================
# These functions handle the RAG pipeline steps.

def get_source_badge(source_type: str) -> str:
    """Return an HTML badge for the source type (PDF/Website/Wikipedia)."""
    if source_type == "PDF":
        return '<span class="source-badge source-pdf">📄 PDF</span>'
    elif source_type == "Website":
        return '<span class="source-badge source-web">🌐 Website</span>'
    elif source_type == "Wikipedia":
        return '<span class="source-badge source-wiki">📚 Wikipedia</span>'
    return f'<span class="source-badge">{source_type}</span>'


def call_llm_with_retry(llm, messages, max_retries=3):
    """
    Call the LLM with automatic retry on rate limit errors.
    
    The Gemini free tier has rate limits (requests per minute).
    If we hit the limit, this function waits and retries automatically
    instead of crashing the app.
    """
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                wait_time = 30  # Wait 30 seconds before retrying
                st.warning(f"⏳ Rate limit hit. Waiting {wait_time}s and retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e  # Re-raise if it's not a rate limit error
    st.error("❌ Rate limit exceeded after multiple retries. Please wait a minute and try again.")
    return None

def save_vectorstore_pkl(vectorstore, folder_path="vector_index.pkl"):
    """
    Save the FAISS vector store to disk.

    WHY SAVE?
    - Next time you open the app, you can load the vectors instantly
      instead of calling the Gemini API again (saves time!).

    NOTE: FAISS uses save_local() which creates a folder with:
      - index.faiss (the vector index)
      - index.pkl (the metadata/documents)
    """
    vectorstore.save_local(folder_path)


def load_vectorstore_pkl(folder_path="vector_index.pkl", embeddings=None):
    """
    Load a previously saved FAISS vector store from disk.
    Returns None if the folder doesn't exist.
    """
    if os.path.exists(folder_path):
        return FAISS.load_local(
            folder_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None


def process_documents(documents, source_label):
    """
    THE CORE RAG PIPELINE — Process documents through the full chain:
    Extract → Chunk → Embed → Store in FAISS → Save as PKL

    Args:
        documents: List of LangChain Document objects (from any loader)
        source_label: Label like "📄 myfile.pdf" for tracking

    Returns:
        True if successful, False otherwise
    """
    try:
        # ---- CHUNKING ----
        # Break long text into smaller pieces (~1000 characters each).
        # WHY? AI has a token limit. We need small, precise chunks for search.
        # Overlap of 200 chars prevents cutting sentences in half.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,       # Max characters per chunk
            chunk_overlap=200,     # Overlap between chunks
            length_function=len,   # Use Python's len() to count
            separators=["\n\n", "\n", " ", ""]  # Split priority order
        )
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            st.error("No text could be extracted. The document might be empty or image-based.")
            return False

        # ---- EMBEDDINGS + FAISS ----
        # Convert each chunk's text into a 1536-dimensional number vector,
        # then store all vectors in the FAISS index for similarity search.
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        if st.session_state.vectorstore is None:
            # First time: create a brand new FAISS vector store
            st.session_state.vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
        else:
            # Already have a store: create a temp one and merge it in
            new_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
            st.session_state.vectorstore.merge_from(new_store)

        # Save chunks for summary/quiz generation later
        st.session_state.all_chunks.extend(chunks)

        # Track the knowledge source
        st.session_state.knowledge_sources.append(source_label)

        # ---- SAVE TO PKL ----
        # Persist the FAISS index to disk as a Pickle file.
        # This way you don't need to re-embed everything next time!
        save_vectorstore_pkl(st.session_state.vectorstore)

        return True

    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False


# ============================================================
# STEP 6: HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🎓 Edualze AI Knowledge Assistant</h1>
    <p>Upload PDFs, add websites, or search Wikipedia — then ask any question!</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# STEP 7: SIDEBAR — Knowledge Source Management
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # --- API Key Status ---
    # Instead of typing the key in the UI (insecure!),
    # we load it securely from the .env file.
    if GOOGLE_API_KEY and GOOGLE_API_KEY != "your-api-key-here":
        st.success("🔑 Gemini API Key loaded from .env")
    else:
        st.error("🔑 API Key missing! Add it to your .env file.")
        st.caption("Open `.env` and set: GOOGLE_API_KEY=AIza...")

    st.markdown("---")

    # =========================================
    # SOURCE 1: PDF UPLOAD
    # =========================================
    # Uses PyPDFLoader to extract text from each page of the PDF.
    # Each page becomes a LangChain Document object.
    st.markdown("## 📚 Add Knowledge Sources")
    st.markdown("### 📄 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload your study materials, notes, or any PDF documents."
    )

    if uploaded_files and st.button("📥 Process PDFs", use_container_width=True):
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your-api-key-here":
            st.error("⚠️ Please add your Google API key to the .env file first!")
        else:
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing: {uploaded_file.name}..."):
                    # Save uploaded file to study_materials/ directory
                    os.makedirs("study_materials", exist_ok=True)
                    file_path = os.path.join("study_materials", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load PDF using PyPDFLoader
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()

                    # Add source labels to metadata
                    for doc in documents:
                        doc.metadata["source_type"] = "PDF"
                        doc.metadata["source_name"] = uploaded_file.name

                    # Process through the RAG pipeline
                    if process_documents(documents, f"📄 {uploaded_file.name}"):
                        st.success(f"✅ Added: {uploaded_file.name}")

    st.markdown("---")

    # =========================================
    # SOURCE 2: WEBSITE URL
    # =========================================
    # Uses WebBaseLoader to fetch and parse webpage HTML into clean text.
    st.markdown("### 🌐 Add Website URL")

    url_input = st.text_input(
        "Paste a website URL",
        placeholder="https://example.com/article",
        help="Enter any article, blog, or webpage URL."
    )

    if st.button("🌐 Add Website", use_container_width=True):
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your-api-key-here":
            st.error("⚠️ Please add your Google API key to the .env file first!")
        elif not url_input:
            st.warning("Please enter a URL first.")
        else:
            with st.spinner(f"Fetching: {url_input}..."):
                try:
                    loader = WebBaseLoader(url_input)
                    documents = loader.load()

                    for doc in documents:
                        doc.metadata["source_type"] = "Website"
                        doc.metadata["source_name"] = url_input

                    if process_documents(documents, f"🌐 {url_input[:50]}..."):
                        st.success("✅ Added website content!")
                except Exception as e:
                    st.error(f"Error loading URL: {str(e)}")

    st.markdown("---")

    # =========================================
    # SOURCE 3: WIKIPEDIA
    # =========================================
    # Uses WikipediaLoader to search and fetch Wikipedia articles.
    st.markdown("### 📚 Search Wikipedia")

    wiki_topic = st.text_input(
        "Enter a Wikipedia topic",
        placeholder="Artificial Intelligence",
        help="Search any topic on Wikipedia."
    )

    if st.button("📚 Add Wikipedia", use_container_width=True):
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your-api-key-here":
            st.error("⚠️ Please add your Google API key to the .env file first!")
        elif not wiki_topic:
            st.warning("Please enter a topic first.")
        else:
            with st.spinner(f"Fetching Wikipedia: {wiki_topic}..."):
                try:
                    loader = WikipediaLoader(query=wiki_topic, load_max_docs=2)
                    documents = loader.load()

                    for doc in documents:
                        doc.metadata["source_type"] = "Wikipedia"
                        doc.metadata["source_name"] = f"Wikipedia: {wiki_topic}"

                    if process_documents(documents, f"📚 Wikipedia: {wiki_topic}"):
                        st.success(f"✅ Added Wikipedia: {wiki_topic}")
                except Exception as e:
                    st.error(f"Error loading Wikipedia: {str(e)}")

    st.markdown("---")

    # =========================================
    # KNOWLEDGE BASE STATUS
    # =========================================
    st.markdown("## 📊 Knowledge Base")
    if st.session_state.knowledge_sources:
        st.markdown('<p class="status-ready">✅ Knowledge base is ready!</p>', unsafe_allow_html=True)
        for source in st.session_state.knowledge_sources:
            st.markdown(f'<div class="knowledge-card">{source}</div>', unsafe_allow_html=True)
        st.markdown(f"**Total chunks:** {len(st.session_state.all_chunks)}")
    else:
        st.markdown('<p class="status-empty">⚠️ No knowledge sources added yet.</p>', unsafe_allow_html=True)
        st.caption("Upload a PDF, add a URL, or search Wikipedia to get started!")

    st.markdown("---")

    # =========================================
    # TOOLS: Summary, Quiz, Clear, Reset
    # =========================================
    st.markdown("## 🛠️ Tools")

    # --- Generate Summary ---
    if st.button("📝 Generate Summary", use_container_width=True):
        if not st.session_state.all_chunks:
            st.warning("Add some knowledge sources first!")
        else:
            with st.spinner("Generating summary..."):
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-lite",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.3
                )
                # Use first 5 chunks for summary (to stay within token limits)
                summary_text = "\n\n".join(
                    [chunk.page_content for chunk in st.session_state.all_chunks[:5]]
                )
                prompt = SUMMARY_PROMPT.format(text=summary_text)
                messages = [
                    SystemMessage(content="You are a helpful study assistant."),
                    HumanMessage(content=prompt)
                ]
                response = call_llm_with_retry(llm, messages)
                if response is None:
                    st.error("Could not generate summary. Try again in a minute.")
                else:
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f"📝 **Document Summary:**\n\n{response.content}"}
                    )
                    st.rerun()

    # --- Generate Quiz ---
    if st.button("❓ Generate Quiz", use_container_width=True):
        if not st.session_state.all_chunks:
            st.warning("Add some knowledge sources first!")
        else:
            with st.spinner("Generating quiz questions..."):
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-lite",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.5
                )
                quiz_text = "\n\n".join(
                    [chunk.page_content for chunk in st.session_state.all_chunks[:5]]
                )
                prompt = QUIZ_PROMPT.format(text=quiz_text)
                messages = [
                    SystemMessage(content="You are a helpful quiz generator for students."),
                    HumanMessage(content=prompt)
                ]
                response = call_llm_with_retry(llm, messages)
                if response is None:
                    st.error("Could not generate quiz. Try again in a minute.")
                else:
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f"❓ **Quiz Time!**\n\n{response.content}"}
                    )
                    st.rerun()

    # --- Clear Chat ---
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # --- Reset Everything ---
    if st.button("🔄 Reset Knowledge Base", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.all_chunks = []
        st.session_state.knowledge_sources = []
        st.session_state.chat_history = []
        # Delete the PKL file if it exists
        if os.path.exists("vector_index.pkl"):
            os.remove("vector_index.pkl")
        st.rerun()


# ============================================================
# STEP 8: MAIN CHAT AREA — The RAG Question-Answering Loop
# ============================================================

# Display existing chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # If the message has sources, show them in an expander
        if "sources" in message:
            with st.expander("📖 View Retrieved Source Chunks"):
                for i, source in enumerate(message["sources"], 1):
                    badge = get_source_badge(source.get("source_type", "Unknown"))
                    st.markdown(
                        f"""<div class="chunk-box">
                            <strong>Chunk {i}</strong> {badge}
                            <br><small>Source: {source.get("source_name", "Unknown")}</small>
                            <hr style="margin: 8px 0;">
                            {source["content"][:500]}{'...' if len(source["content"]) > 500 else ''}
                        </div>""",
                        unsafe_allow_html=True
                    )

# Chat input — where the user types their question
user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    # Validate: API key must be set
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your-api-key-here":
        st.error("⚠️ Please add your Google API key to the .env file!")

    # Validate: Knowledge base must exist
    elif st.session_state.vectorstore is None:
        st.warning("📚 Please add some knowledge sources first! Use the sidebar.")

    else:
        # Display user's message
        with st.chat_message("user"):
            st.markdown(user_question)

        # Add user message to history
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )

        # Generate the AI answer using RAG
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):

                # ---- RETRIEVAL: Find relevant chunks ----
                # This is the "R" in RAG!
                # FAISS converts the question to a vector,
                # then finds the 4 chunks with the closest vectors.
                retrieved_chunks = st.session_state.vectorstore.similarity_search(
                    user_question, k=4
                )

                # ---- BUILD CONTEXT: Combine retrieved chunks ----
                context = "\n\n---\n\n".join(
                    [chunk.page_content for chunk in retrieved_chunks]
                )

                # ---- FORMAT CHAT HISTORY ----
                # Include last 6 messages so AI remembers context
                recent_history = st.session_state.chat_history[-6:]
                chat_history_str = ""
                for msg in recent_history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    chat_history_str += f"{role}: {msg['content']}\n"

                # ---- LLM: Generate the answer ----
                # Send context + question to GPT-3.5-turbo
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-lite",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.3   # Low = factual, High = creative
                )

                prompt = QA_PROMPT.format(
                    context=context,
                    chat_history=chat_history_str,
                    question=user_question
                )

                messages = [
                    SystemMessage(content="You are Edualze AI, a helpful study assistant."),
                    HumanMessage(content=prompt)
                ]

                response = call_llm_with_retry(llm, messages)
                if response is None:
                    st.stop()
                answer = response.content

                # Display the answer
                st.markdown(answer)

                # ---- SHOW SOURCE CHUNKS ----
                # Display which chunks were used, with source labels
                sources_data = []
                with st.expander("📖 View Retrieved Source Chunks"):
                    for i, chunk in enumerate(retrieved_chunks, 1):
                        source_type = chunk.metadata.get("source_type", "Unknown")
                        source_name = chunk.metadata.get("source_name", "Unknown")
                        badge = get_source_badge(source_type)

                        st.markdown(
                            f"""<div class="chunk-box">
                                <strong>Chunk {i}</strong> {badge}
                                <br><small>Source: {source_name}</small>
                                <hr style="margin: 8px 0;">
                                {chunk.page_content[:500]}{'...' if len(chunk.page_content) > 500 else ''}
                            </div>""",
                            unsafe_allow_html=True
                        )

                        sources_data.append({
                            "content": chunk.page_content,
                            "source_type": source_type,
                            "source_name": source_name
                        })

                # Save to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_data
                })


# ============================================================
# STEP 9: FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """<div style="text-align: center; color: #94a3b8; font-size: 0.8rem;">
        Built with ❤️ using Streamlit, LangChain, FAISS & Google Gemini |
        <strong>Edualze AI Knowledge Assistant</strong>
    </div>""",
    unsafe_allow_html=True
)
