"""
LexAssist — Streamlit Chat UI
Run: streamlit run app.py
Make sure to run ingest.py first.
"""

import os, time, torch
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "lexassist"
EMBED_MODEL     = "BAAI/bge-large-en-v1.5"
GROQ_MODEL      = "openai/gpt-oss-120b"
RETRIEVER_K     = 4

SYSTEM_PROMPT = """You are LexAssist, an expert Indian legal and government scheme assistant.
Answer questions STRICTLY from the provided context.
If the context does not contain enough information, say: "I don't have enough information in the provided documents to answer this."
Always mention which document/section your answer is based on.
Be concise, accurate, and use plain language."""

ARCH_SVG = """<svg width="100%" viewBox="0 0 680 430" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<style>
.box{fill:#1e293b;stroke:#334155;stroke-width:0.5px;rx:8;}
.teal{fill:#0F6E56;stroke:#1D9E75;}
.gray{fill:#3d3d3a;stroke:#5F5E5A;}
.amber{fill:#854F0B;stroke:#BA7517;}
.th{font-family:sans-serif;font-size:13px;font-weight:500;fill:#e2e8f0;text-anchor:middle;dominant-baseline:central;}
.ts{font-family:sans-serif;font-size:11px;fill:#94a3b8;text-anchor:middle;dominant-baseline:central;}
.arr-line{stroke:#1D9E75;stroke-width:0.8;stroke-dasharray:4 3;fill:none;}
</style>
<text class="ts" x="340" y="15" style="fill:#64748b;">Indexing pipeline (offline)</text>
<rect x="30" y="24" width="138" height="54" rx="8" class="teal" stroke-width="0.5"/>
<text class="th" x="99" y="43">Raw docs</text><text class="ts" x="99" y="61">IPC · RTI · 4 PDFs</text>
<line x1="168" y1="51" x2="188" y2="51" stroke="#1D9E75" stroke-width="1" marker-end="url(#arr)"/>
<rect x="188" y="24" width="138" height="54" rx="8" class="teal" stroke-width="0.5"/>
<text class="th" x="257" y="43">Load &amp; chunk</text><text class="ts" x="257" y="61">PyPDF · 1712 chunks</text>
<line x1="326" y1="51" x2="346" y2="51" stroke="#1D9E75" stroke-width="1" marker-end="url(#arr)"/>
<rect x="346" y="24" width="138" height="54" rx="8" class="teal" stroke-width="0.5"/>
<text class="th" x="415" y="43">BGE embeddings</text><text class="ts" x="415" y="61">bge-large · CUDA</text>
<line x1="484" y1="51" x2="504" y2="51" stroke="#1D9E75" stroke-width="1" marker-end="url(#arr)"/>
<rect x="504" y="24" width="138" height="54" rx="8" class="teal" stroke-width="0.5"/>
<text class="th" x="573" y="43">ChromaDB</text><text class="ts" x="573" y="61">1712 vectors · disk</text>
<path d="M573,78 L573,110 L257,110 L257,116" class="arr-line" marker-end="url(#arr)"/>
<text class="ts" x="340" y="106" style="fill:#64748b;">Query pipeline (online)</text>
<rect x="30" y="116" width="138" height="54" rx="8" class="teal" stroke-width="0.5"/>
<text class="th" x="99" y="135">User query</text><text class="ts" x="99" y="153">Legal questions</text>
<line x1="168" y1="143" x2="188" y2="143" stroke="#1D9E75" stroke-width="1" marker-end="url(#arr)"/>
<rect x="188" y="116" width="138" height="54" rx="8" class="gray" stroke-width="0.5"/>
<text class="th" x="257" y="135">Query rewrite</text><text class="ts" x="257" y="153">HyDE · multi-query</text>
<line x1="326" y1="143" x2="346" y2="143" stroke="#5F5E5A" stroke-width="1" marker-end="url(#arr)"/>
<rect x="346" y="116" width="138" height="54" rx="8" class="teal" stroke-width="0.5"/>
<text class="th" x="415" y="135">MMR retriever</text><text class="ts" x="415" y="153">k=4 · fetch_k=10</text>
<line x1="484" y1="143" x2="504" y2="143" stroke="#1D9E75" stroke-width="1" marker-end="url(#arr)"/>
<rect x="504" y="116" width="138" height="54" rx="8" class="gray" stroke-width="0.5"/>
<text class="th" x="573" y="135">Reranker</text><text class="ts" x="573" y="153">Cross-encoder</text>
<path d="M415,170 L415,200" class="arr-line" marker-end="url(#arr)"/>
<text class="ts" x="340" y="196" style="fill:#64748b;">Generation</text>
<rect x="96" y="205" width="138" height="54" rx="8" class="gray" stroke-width="0.5"/>
<text class="th" x="165" y="224">Ctx compress</text><text class="ts" x="165" y="242">Not in app.py</text>
<line x1="234" y1="232" x2="264" y2="232" stroke="#5F5E5A" stroke-width="1" marker-end="url(#arr)"/>
<rect x="264" y="205" width="152" height="54" rx="8" class="teal" stroke-width="0.5"/>
<text class="th" x="340" y="224">Groq LLM</text><text class="ts" x="340" y="242">llama-3.1-70b</text>
<line x1="416" y1="232" x2="446" y2="232" stroke="#1D9E75" stroke-width="1" marker-end="url(#arr)"/>
<rect x="446" y="205" width="138" height="54" rx="8" class="teal" stroke-width="0.5"/>
<text class="th" x="515" y="224">Final answer</text><text class="ts" x="515" y="242">With citations</text>
<path d="M340,259 L340,290" class="arr-line" marker-end="url(#arr)"/>
<text class="ts" x="340" y="286" style="fill:#64748b;">Production &amp; evaluation</text>
<rect x="96" y="295" width="138" height="54" rx="8" class="teal" stroke-width="0.5"/>
<text class="th" x="165" y="314">Streamlit UI</text><text class="ts" x="165" y="332">Dark theme chat</text>
<rect x="264" y="295" width="152" height="54" rx="8" class="teal" stroke-width="0.5"/>
<text class="th" x="340" y="314">Chat memory</text><text class="ts" x="340" y="332">Session history</text>
<rect x="446" y="295" width="138" height="54" rx="8" class="amber" stroke-width="0.5"/>
<text class="th" x="515" y="314">RAGAS eval</text><text class="ts" x="515" y="332">Scaffold ready</text>
<rect x="96" y="376" width="12" height="12" rx="2" fill="#0F6E56" stroke="#1D9E75" stroke-width="0.5"/>
<text class="ts" x="114" y="382" style="text-anchor:start;">Implemented</text>
<rect x="220" y="376" width="12" height="12" rx="2" fill="#3d3d3a" stroke="#5F5E5A" stroke-width="0.5"/>
<text class="ts" x="238" y="382" style="text-anchor:start;">Not in app.py</text>
<rect x="344" y="376" width="12" height="12" rx="2" fill="#854F0B" stroke="#BA7517" stroke-width="0.5"/>
<text class="ts" x="362" y="382" style="text-anchor:start;">Partial</text>
</svg>"""

st.set_page_config(page_title="LexAssist", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0d14; }
section[data-testid="stSidebar"] { background: #13131f; border-right: 1px solid #2a2a3d; }
.main-header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); padding: 28px 32px; border-radius: 14px; margin-bottom: 24px; border: 1px solid #2a2a5a; }
.main-header h1 { color: #e2e8f0; font-size: 2em; font-weight: 600; margin: 0 0 4px 0; }
.main-header p  { color: #94a3b8; margin: 0; font-size: 0.95em; }
.chat-bubble-user { background: #1e293b; border: 1px solid #334155; border-radius: 12px 12px 4px 12px; padding: 14px 18px; margin: 8px 0; color: #e2e8f0; font-size: 0.95em; }
.chat-bubble-bot  { background: #131929; border: 1px solid #1e3a5f; border-radius: 12px 12px 12px 4px; padding: 14px 18px; margin: 8px 0; color: #e2e8f0; font-size: 0.95em; line-height: 1.7; }
.source-card { background: #0f1923; border: 1px solid #1e3050; border-left: 3px solid #3b82f6; border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 0.82em; color: #94a3b8; font-family: 'DM Mono', monospace; }
.source-card strong { color: #60a5fa; font-family: 'DM Sans', sans-serif; }
.metric-pill { display: inline-block; background: #1e293b; border: 1px solid #334155; border-radius: 20px; padding: 3px 12px; font-size: 0.78em; color: #94a3b8; margin-right: 8px; }
.metric-pill span { color: #60a5fa; font-weight: 600; }
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.status-dot.green { background: #10b981; }
.status-dot.red   { background: #ef4444; }
.sidebar-section { background: #1a1a2e; border-radius: 10px; padding: 14px; margin-bottom: 14px; border: 1px solid #2a2a4a; }
.sidebar-section h4 { color: #a78bfa; font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 10px 0; }
</style>
""", unsafe_allow_html=True)

for k, v in {"messages": [], "chain_ready": False, "session_id": "user_01",
             "session_store": {}, "total_queries": 0, "avg_latency": 0.0,
             "show_arch": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    if not Path(CHROMA_DIR).exists() or not any(Path(CHROMA_DIR).iterdir()):
        return None, None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    llm = ChatGroq(
        model=GROQ_MODEL, temperature=0.1, max_tokens=1024,
        api_key=os.getenv("GROQ_API_KEY")
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": 10}
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Source {i+1}: {Path(d.metadata.get('source','?')).name}, page {d.metadata.get('page','?')}]\n{d.page_content}"
            for i, d in enumerate(docs)
        )

    conv_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

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

    groq_ok  = bool(os.getenv("GROQ_API_KEY"))
    data_ok  = Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir())
    gpu_ok   = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_ok else "CPU mode"

    st.markdown('<div class="sidebar-section"><h4>System Status</h4>', unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:0.88em;color:#cbd5e1;'>"
        f"<span class='status-dot {'green' if groq_ok else 'red'}'></span>Groq API<br>"
        f"<span class='status-dot {'green' if data_ok else 'red'}'></span>Vectorstore<br>"
        f"<span class='status-dot {'green' if gpu_ok else 'red'}'></span>{gpu_name}"
        f"</p></div>", unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section"><h4>Model Info</h4>', unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:0.85em;color:#94a3b8;'>"
        f"LLM: <b style='color:#a78bfa;'>{GROQ_MODEL}</b><br>"
        f"Embed: <b style='color:#a78bfa;'>BGE-large-en-v1.5</b><br>"
        f"Retriever: <b style='color:#a78bfa;'>MMR k={RETRIEVER_K}</b>"
        f"</p></div>", unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section"><h4>Session Stats</h4>', unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:0.88em;color:#94a3b8;'>"
        f"Queries: <b style='color:#60a5fa;'>{st.session_state.total_queries}</b><br>"
        f"Avg latency: <b style='color:#60a5fa;'>{st.session_state.avg_latency:.1f}s</b>"
        f"</p></div>", unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section"><h4>Try Asking</h4>', unsafe_allow_html=True)
    for s in ["What are fundamental rights?", "What is IPC Section 302?",
              "Who can file an RTI?", "What is the grant scheme?",
              "Who is eligible to apply?"]:
        if st.button(s, use_container_width=True, key=f"s_{s[:15]}"):
            st.session_state["inject_question"] = s
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗺️ Show architecture", use_container_width=True):
        st.session_state.show_arch = not st.session_state.show_arch
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.session_store = {}
        st.session_state.total_queries = 0
        st.session_state.avg_latency   = 0.0
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>⚖️ LexAssist</h1>
  <p>Indian Legal &amp; Government Schemes Navigator &nbsp;·&nbsp; llama-3.1-70b via Groq &nbsp;·&nbsp; BGE-large on GPU &nbsp;·&nbsp; RAG</p>
</div>
""", unsafe_allow_html=True)

# Architecture diagram toggle
if st.session_state.show_arch:
    st.markdown("#### RAG Architecture — what's implemented")
    st.markdown(ARCH_SVG, unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.8em;color:#475569;margin-top:6px;'>"
        "Teal = implemented &nbsp;|&nbsp; Gray = notebook only &nbsp;|&nbsp; Amber = partial"
        "</p>", unsafe_allow_html=True
    )
    st.markdown("---")

# Load pipeline
if not st.session_state.chain_ready:
    with st.spinner("⚙️ Loading RAG pipeline..."):
        chain_base, retriever, vectorstore = load_rag_pipeline()
        if chain_base is None:
            st.error("⚠️ Vectorstore not found. Run this first:")
            st.code("python ingest.py", language="bash")
            st.stop()
        st.session_state.chain_base  = chain_base
        st.session_state.retriever   = retriever
        st.session_state.vectorstore = vectorstore
        st.session_state.chain_ready = True


def get_session_history(sid):
    if sid not in st.session_state.session_store:
        st.session_state.session_store[sid] = ChatMessageHistory()
    return st.session_state.session_store[sid]


conv_rag = RunnableWithMessageHistory(
    st.session_state.chain_base,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-bot">⚖️ {msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s)", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card"><strong>{src["file"]}</strong> · page {src["page"]}<br>{src["text"]}...</div>',
                        unsafe_allow_html=True
                    )
        if "latency" in msg:
            st.markdown(
                f'<div class="metric-pill">⏱ <span>{msg["latency"]}s</span></div>'
                f'<div class="metric-pill">🔍 <span>{msg.get("chunks",0)} chunks</span></div>',
                unsafe_allow_html=True
            )

# Chat input
injected   = st.session_state.pop("inject_question", None)
user_input = st.chat_input("Ask about Indian law, IPC, Constitution, government schemes...")
question   = injected or user_input

if question:
    st.markdown(f'<div class="chat-bubble-user">👤 {question}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("🔍 Retrieving & generating answer..."):
        t0 = time.time()
        try:
            answer  = conv_rag.invoke(
                {"question": question},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            latency = round(time.time() - t0, 2)
            sources_raw = st.session_state.retriever.invoke(question)
            sources = [
                {"file": Path(d.metadata.get("source","?")).name,
                 "page": d.metadata.get("page","?"),
                 "text": d.page_content[:280].strip()}
                for d in sources_raw
            ]
            n = st.session_state.total_queries + 1
            st.session_state.avg_latency   = (st.session_state.avg_latency * (n-1) + latency) / n
            st.session_state.total_queries = n
            st.session_state.messages.append({
                "role": "assistant", "content": answer,
                "sources": sources, "latency": latency, "chunks": len(sources)
            })
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#475569;">
      <div style="font-size:3em;margin-bottom:16px;">⚖️</div>
      <p style="font-size:1.1em;color:#64748b;">Ask any question about your legal documents</p>
      <p style="font-size:0.85em;color:#475569;">Try the suggestions in the sidebar or click <b style="color:#a78bfa;">Show architecture</b> to see how RAG works</p>
    </div>
    """, unsafe_allow_html=True)