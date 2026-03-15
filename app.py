"""
LexAssist — Streamlit Chat UI
Run: streamlit run app.py
"""

import os, time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

DATA_DIR        = "Data"
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "lexassist"
EMBED_MODEL     = "BAAI/bge-large-en-v1.5"
GROQ_MODEL      = "openai/gpt-oss-120b"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
RETRIEVER_K     = 4

SYSTEM_PROMPT = """You are LexAssist, an expert Indian legal and government scheme assistant.
Answer questions STRICTLY from the provided context.
If the context does not contain enough information, say: "I don't have enough information in the provided documents to answer this."
Always mention which document/section your answer is based on.
Be concise, accurate, and use plain language."""

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexAssist",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0d0d14;
}

section[data-testid="stSidebar"] {
    background: #13131f;
    border-right: 1px solid #2a2a3d;
}

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 28px 32px;
    border-radius: 14px;
    margin-bottom: 24px;
    border: 1px solid #2a2a5a;
}
.main-header h1 {
    color: #e2e8f0;
    font-size: 2em;
    font-weight: 600;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #94a3b8;
    margin: 0;
    font-size: 0.95em;
}

.chat-bubble-user {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px 12px 4px 12px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #e2e8f0;
    font-size: 0.95em;
}
.chat-bubble-bot {
    background: #131929;
    border: 1px solid #1e3a5f;
    border-radius: 12px 12px 12px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #e2e8f0;
    font-size: 0.95em;
    line-height: 1.7;
}

.source-card {
    background: #0f1923;
    border: 1px solid #1e3050;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.82em;
    color: #94a3b8;
    font-family: 'DM Mono', monospace;
}
.source-card strong {
    color: #60a5fa;
    font-family: 'DM Sans', sans-serif;
}

.metric-pill {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78em;
    color: #94a3b8;
    margin-right: 8px;
}
.metric-pill span {
    color: #60a5fa;
    font-weight: 600;
}

.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot.green { background: #10b981; }
.status-dot.red   { background: #ef4444; }

.sidebar-section {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 14px;
    border: 1px solid #2a2a4a;
}
.sidebar-section h4 {
    color: #a78bfa;
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages = []
if "chain_ready"  not in st.session_state: st.session_state.chain_ready = False
if "session_id"   not in st.session_state: st.session_state.session_id = "user_01"
if "session_store" not in st.session_state: st.session_state.session_store = {}
if "use_advanced" not in st.session_state: st.session_state.use_advanced = False
if "total_queries" not in st.session_state: st.session_state.total_queries = 0
if "avg_latency"   not in st.session_state: st.session_state.avg_latency = 0.0


# ── RAG Pipeline (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    """Load RAG pipeline from pre-built vectorstore. Run ingest.py first."""

    # ── Guard: chroma_db must exist
    if not Path(CHROMA_DIR).exists() or not any(Path(CHROMA_DIR).iterdir()):
        st.error("⚠️ Vectorstore not found. Run this first in your terminal:")
        st.code("python ingest.py", language="bash")
        st.stop()

    # ── Embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBED_MODEL,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    # ── Load vectorstore from disk (instant — no embedding)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # ── LLM
    llm = ChatGroq(
        model=GROQ_MODEL, temperature=0.1, max_tokens=1024,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # ── Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": 10}
    )

    def format_docs(docs):
        parts = []
        for i, doc in enumerate(docs, 1):
            source = Path(doc.metadata.get("source", "unknown")).name
            page   = doc.metadata.get("page", "?")
            parts.append(f"[Source {i}: {source}, page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    # ── Conversational prompt
    conv_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    # ── Chain
    chain_base = (
        RunnableParallel({
            "context"     : lambda x: format_docs(retriever.invoke(x["question"])),
            "question"    : lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", [])
        })
        | conv_prompt | llm | StrOutputParser()
    )

    return chain_base, retriever, vectorstore


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ LexAssist")
    st.markdown("---")

    # Status
    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    hf_ok   = bool(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    data_ok = Path(DATA_DIR).exists() and bool(list(Path(DATA_DIR).glob("*.pdf")))

    st.markdown("""<div class="sidebar-section"><h4>System Status</h4>""", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:0.88em; color:#cbd5e1;'>"
        f"<span class='status-dot {'green' if groq_ok else 'red'}'></span>Groq API<br>"
        f"<span class='status-dot {'green' if hf_ok else 'red'}'></span>HuggingFace API<br>"
        f"<span class='status-dot {'green' if data_ok else 'red'}'></span>PDF documents</p>"
        "</div>", unsafe_allow_html=True
    )

    # Settings
    st.markdown("""<div class="sidebar-section"><h4>Retrieval Settings</h4>""", unsafe_allow_html=True)
    use_adv = st.toggle("Contextual compression", value=False,
                        help="Strips irrelevant sentences from retrieved chunks. Slower but more precise.")
    k_val   = st.slider("Chunks to retrieve (k)", 2, 8, RETRIEVER_K)
    st.markdown("</div>", unsafe_allow_html=True)

    # Stats
    st.markdown("""<div class="sidebar-section"><h4>Session Stats</h4>""", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:0.88em; color:#94a3b8;'>"
        f"Queries: <b style='color:#60a5fa;'>{st.session_state.total_queries}</b><br>"
        f"Avg latency: <b style='color:#60a5fa;'>{st.session_state.avg_latency:.1f}s</b>"
        f"</p></div>", unsafe_allow_html=True
    )

    # Suggested questions
    st.markdown("""<div class="sidebar-section"><h4>Try Asking</h4>""", unsafe_allow_html=True)
    suggestions = [
        "What is the grant scheme?",
        "Who is eligible to apply?",
        "What are the conditions for using the grant?",
        "What happens if conditions are violated?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sugg_{s[:20]}"):
            st.session_state["inject_question"] = s
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_store = {}
        st.session_state.total_queries = 0
        st.session_state.avg_latency   = 0.0
        st.rerun()


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>⚖️ LexAssist</h1>
  <p>Indian Legal & Government Schemes Navigator · Powered by RAG + Groq + BGE</p>
</div>
""", unsafe_allow_html=True)

# Load pipeline
if not st.session_state.chain_ready:
    with st.spinner("⚙️ Loading RAG pipeline..."):
        try:
            chain_base, retriever, vectorstore = load_rag_pipeline()
            st.session_state.chain_base   = chain_base
            st.session_state.retriever    = retriever
            st.session_state.vectorstore  = vectorstore
            st.session_state.chain_ready  = True
        except Exception as e:
            st.error(f"Failed to load pipeline: {e}")
            st.stop()

# Session history getter
def get_session_history(sid: str) -> ChatMessageHistory:
    if sid not in st.session_state.session_store:
        st.session_state.session_store[sid] = ChatMessageHistory()
    return st.session_state.session_store[sid]

# Wrap chain with history
conv_rag = RunnableWithMessageHistory(
    st.session_state.chain_base,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# ── Chat History ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user">👤 {msg["content"]}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-bot">⚖️ {msg["content"]}</div>',
                    unsafe_allow_html=True)
        # Show sources
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📎 {len(msg['sources'])} source(s) used", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>{src["file"]}</strong> · page {src["page"]}<br>'
                        f'{src["text"]}...'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        # Metrics
        if "latency" in msg:
            st.markdown(
                f'<div class="metric-pill">⏱ <span>{msg["latency"]}s</span></div>'
                f'<div class="metric-pill">🔍 <span>{msg.get("chunks",0)} chunks</span></div>',
                unsafe_allow_html=True
            )

# ── Chat Input ────────────────────────────────────────────────────────────────
injected = st.session_state.pop("inject_question", None)
user_input = st.chat_input("Ask about Indian law, IPC, Constitution, government schemes...")
question = injected or user_input

if question:
    # Show user message
    st.markdown(f'<div class="chat-bubble-user">👤 {question}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("🔍 Retrieving context & generating answer..."):
        t0 = time.time()
        try:
            answer = conv_rag.invoke(
                {"question": question},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            latency = round(time.time() - t0, 2)

            # Retrieve sources
            sources_raw = st.session_state.retriever.invoke(question)
            sources = [
                {
                    "file": Path(d.metadata.get("source", "unknown")).name,
                    "page": d.metadata.get("page", "?"),
                    "text": d.page_content[:280].strip()
                }
                for d in sources_raw
            ]

            # Update stats
            n = st.session_state.total_queries + 1
            st.session_state.avg_latency   = (st.session_state.avg_latency * (n-1) + latency) / n
            st.session_state.total_queries = n

            # Save to history
            st.session_state.messages.append({
                "role"   : "assistant",
                "content": answer,
                "sources": sources,
                "latency": latency,
                "chunks" : len(sources)
            })

            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #475569;">
      <div style="font-size: 3em; margin-bottom: 16px;">⚖️</div>
      <p style="font-size: 1.1em; color: #64748b;">Ask any question about your legal documents</p>
      <p style="font-size: 0.85em; color: #475569;">Try the suggestions in the sidebar →</p>
    </div>
    """, unsafe_allow_html=True)
