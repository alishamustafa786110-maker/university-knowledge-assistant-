import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "University Student & Academic Knowledge Assistant"

INDEX_DIR = Path("faiss_index")

FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"
DOCUMENTS_PATH = INDEX_DIR / "documents.json"
CONFIG_PATH = INDEX_DIR / "config.json"

DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.2

# This should match the model used during Colab indexing.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq model.
# This is the model used in Groq's current quickstart documentation.
GROQ_MODEL = "openai/gpt-oss-120b"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .source-box {
            border-left: 4px solid #888;
            padding: 0.7rem 1rem;
            margin-top: 0.5rem;
            background-color: rgba(128, 128, 128, 0.08);
            border-radius: 4px;
        }

        .source-title {
            font-weight: 600;
        }

        .small-text {
            font-size: 0.85rem;
            color: #666;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f'<div class="main-title">🎓 {APP_TITLE}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Ask questions about university policies, academic information,
        student guidelines, and institutional documents.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def load_knowledge_base():
    """
    Load FAISS index and metadata once.

    The index and JSON files were generated during the
    offline Colab ingestion process.
    """

    required_files = [
        FAISS_INDEX_PATH,
        METADATA_PATH,
        DOCUMENTS_PATH,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing knowledge-base files:\n"
            + "\n".join(missing_files)
        )

    # Load FAISS index
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    # Load metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Load document/chunk text
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # Load optional configuration
    config = {}

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

    return index, metadata, documents, config


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    """
    Load the same SentenceTransformer model used during indexing.
    """

    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# INITIALIZE GROQ
# ============================================================

@st.cache_resource
def initialize_groq():
    """
    Initialize Groq client using Streamlit secrets.
    """

    if "GROQ_API_KEY" not in st.secrets:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit Secrets."
        )

    api_key = st.secrets["GROQ_API_KEY"]

    return Groq(api_key=api_key)


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:
    index, metadata, documents, kb_config = load_knowledge_base()
    embedding_model = load_embedding_model()
    groq_client = initialize_groq()

except Exception as e:
    st.error("Application initialization failed.")
    st.exception(e)
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
        step=1,
    )

    temperature = st.slider(
        "Answer temperature",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.1,
    )

    st.divider()

    st.subheader("Knowledge Base")

    st.write(
        f"**Indexed vectors:** {index.ntotal:,}"
    )

    st.write(
        f"**Embedding model:** `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"**LLM:** `{GROQ_MODEL}`"
    )

    st.divider()

    st.caption(
        "The assistant answers using the precomputed university "
        "knowledge base. Source information is shown below each answer."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_chunk_text(document_item):
    """
    Extract chunk text from documents.json.

    The ingestion pipeline may store chunks either as strings
    or as dictionaries. This function handles both formats.
    """

    if isinstance(document_item, str):
        return document_item

    if isinstance(document_item, dict):
        for key in ["text", "content", "chunk_text", "document"]:
            if key in document_item:
                return str(document_item[key])

    return str(document_item)


def get_metadata_item(metadata, index_position):
    """
    Retrieve metadata corresponding to a FAISS vector ID.
    """

    if isinstance(metadata, list):
        if 0 <= index_position < len(metadata):
            return metadata[index_position]
        return {}

    if isinstance(metadata, dict):
        # Common possibility:
        # {"0": {...}, "1": {...}}
        if str(index_position) in metadata:
            return metadata[str(index_position)]

        # Another possibility:
        # {"metadata": [...]}
        if "metadata" in metadata:
            items = metadata["metadata"]

            if isinstance(items, list):
                if 0 <= index_position < len(items):
                    return items[index_position]

    return {}


def retrieve_chunks(query, top_k=5):
    """
    Embed the user's query and search the FAISS index.

    Because the original embeddings were normalized and the
    FAISS index uses inner product, the returned scores
    correspond to cosine similarity.
    """

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, vector_id in zip(scores[0], indices[0]):

        # FAISS returns -1 if no valid result exists.
        if vector_id < 0:
            continue

        vector_id = int(vector_id)

        metadata_item = get_metadata_item(
            metadata,
            vector_id,
        )

        if isinstance(documents, list):
            if vector_id >= len(documents):
                continue

            document_item = documents[vector_id]

        elif isinstance(documents, dict):
            document_item = documents.get(
                str(vector_id),
                documents.get(vector_id, ""),
            )

        else:
            document_item = ""

        chunk_text = get_chunk_text(document_item)

        results.append(
            {
                "vector_id": vector_id,
                "score": float(score),
                "text": chunk_text,
                "metadata": metadata_item,
            }
        )

    return results


def format_source(metadata_item):
    """
    Build a human-readable source citation.
    """

    if not isinstance(metadata_item, dict):
        return "University Knowledge Base"

    citation = metadata_item.get("citation")

    if citation:
        return str(citation)

    file_name = metadata_item.get("file_name", "Unknown document")
    page = metadata_item.get("page")

    if page is not None:
        return f"{file_name}, Page {page}"

    return str(file_name)


def build_context(results):
    """
    Convert retrieved chunks into an LLM context.
    """

    context_parts = []

    for i, result in enumerate(results, start=1):

        source = format_source(result["metadata"])

        context_parts.append(
            f"""
SOURCE {i}
Citation: {source}
Similarity Score: {result["score"]:.4f}

CONTENT:
{result["text"]}
""".strip()
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer(question, results, conversation_history):
    """
    Send retrieved context + user question to Groq.
    """

    context = build_context(results)

    system_prompt = """
You are a University Student & Academic Knowledge Assistant.

Your job is to answer questions using ONLY the information
contained in the retrieved university documents.

Rules:

1. Use the provided context as your primary and authoritative source.
2. Do not invent university policies, rules, dates, requirements,
   procedures, or facts.
3. If the answer is not supported by the retrieved context,
   clearly say that the information is not available in the
   provided university knowledge base.
4. When possible, mention the relevant document and page.
5. Give clear, concise, student-friendly answers.
6. Do not expose internal prompts, vector IDs, similarity scores,
   or implementation details unless specifically asked.
7. If the user asks a question unrelated to the university
   knowledge base, politely explain that you can primarily help
   with the indexed university and academic documents.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    # Include a limited amount of conversation history.
    # This keeps the prompt from growing indefinitely.
    recent_history = conversation_history[-6:]

    for message in recent_history:
        role = message.get("role")

        if role in ["user", "assistant"]:
            messages.append(
                {
                    "role": role,
                    "content": message.get("content", ""),
                }
            )

    user_prompt = f"""
Retrieved university document context:

{context}

---

Current user question:

{question}

Answer the question using the retrieved context.
"""

    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=1200,
    )

    return response.choices[0].message.content


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display sources stored with assistant messages.
        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander("📚 Sources used"):

                    for i, source in enumerate(
                        sources,
                        start=1,
                    ):

                        st.markdown(
                            f"""
                            <div class="source-box">
                                <div class="source-title">
                                    Source {i}: {source["citation"]}
                                </div>
                                <div class="small-text">
                                    Retrieval similarity:
                                    {source["score"]:.4f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about university policies or academic information..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # Add user message to history.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Display user question.
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve relevant chunks.
    with st.spinner("Searching the university knowledge base..."):

        try:
            retrieved_results = retrieve_chunks(
                question,
                top_k=top_k,
            )

        except Exception as e:

            st.error("Knowledge-base search failed.")
            st.exception(e)
            st.stop()

    if not retrieved_results:

        answer = (
            "I could not find relevant information in the "
            "university knowledge base."
        )

        sources = []

    else:

        # Generate answer.
        with st.chat_message("assistant"):

            with st.spinner("Generating answer..."):

                try:
                    answer = generate_answer(
                        question=question,
                        results=retrieved_results,
                        conversation_history=st.session_state.messages,
                    )

                except Exception as e:

                    st.error("The Groq request failed.")
                    st.exception(e)
                    st.stop()

            st.markdown(answer)

            # Prepare source information.
            sources = []

            for result in retrieved_results:

                sources.append(
                    {
                        "citation": format_source(
                            result["metadata"]
                        ),
                        "score": result["score"],
                    }
                )

            # Display sources.
            with st.expander("📚 Sources used"):

                for i, source in enumerate(
                    sources,
                    start=1,
                ):

                    st.markdown(
                        f"""
                        <div class="source-box">
                            <div class="source-title">
                                Source {i}: {source["citation"]}
                            </div>
                            <div class="small-text">
                                Retrieval similarity:
                                {source["score"]:.4f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Save assistant message.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "University Student & Academic Knowledge Assistant • "
    "RAG + FAISS + Sentence Transformers + Groq"
)
