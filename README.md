# ⚖️ LexAssist — Indian Legal & Government Schemes RAG System

> An Advanced Retrieval-Augmented Generation (RAG) system for navigating Indian legal documents, IPC sections, Constitutional articles, RTI procedures, and government schemes — powered by LangChain, ChromaDB, BGE Embeddings, and Groq LLM.

---

## 📸 Preview

```
User: What are the fundamental rights of citizens?
LexAssist: According to the Constitution of India (Part III, Articles 12-35),
           the fundamental rights include Right to Equality (Art. 14-18),
           Right to Freedom (Art. 19-22)...
           [Source: constitution.pdf, page 14]
```

---

## 🗺️ Project Structure

```
RAG_PROJECT/
│
├── Data/                        ← Put your PDFs here
│   ├── constitution.pdf
│   ├── ipc.pdf
│   ├── rti.pdf
│   └── consumer.pdf
│
├── chroma_db/                   ← Auto-generated vector store (run ingest.py)
│
├── collecting_data.py           ← Download legal PDFs automatically
├── ingest.py                    ← Build ChromaDB vectorstore (run ONCE)
├── app.py                       ← Streamlit chat UI
├── lexassist_rag.ipynb          ← Full RAG pipeline notebook (learning)
├── .env                         ← Your API keys (never commit this)
├── .env.example                 ← Template for .env
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment
```bash
git clone https://github.com/yourusername/RAG_PROJECT.git
cd RAG_PROJECT
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp .env.example .env  # Copy template
# Edit .env with your actual API keys
```

### 3. Add Legal Documents
```bash
# Place PDFs in Data/ folder, or run:
python collecting_data.py  # Auto-download legal docs
```

### 4. Build Vector Store
```bash
python ingest.py  # Run once to create chroma_db/
```

### 5. Launch Chat UI
```bash
streamlit run app.py
```

---

## 🔑 API Keys Required

| Stage | Component | Tech Used |
|-------|-----------|-----------|
| 1 | Document Loading | `DirectoryLoader`, `PyPDFLoader` |
| 2 | Text Chunking | `RecursiveCharacterTextSplitter` |
| 3 | Embeddings | `BAAI/bge-large-en-v1.5` (local GPU) |
| 4 | Vector Store | `ChromaDB` (persisted to disk) |
| 5 | Basic RAG Chain | LangChain LCEL pipeline |
| 6 | Advanced Retrieval | HyDE, Multi-Query, Contextual Compression |
| 7 | Conversational Memory | `RunnableWithMessageHistory` |
| 8 | Evaluation | RAGAS (Faithfulness, Relevancy, Precision, Recall) |
| 9 | Chat UI | Streamlit |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU (recommended) — RTX 3060 or higher
- 8GB+ RAM

### Step 1 — Clone / Download the project
```bash
cd your-project-folder
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirment.txt
```

### Step 4 — Configure API Keys
```bash
cp .env
```

Open `.env` and fill in:
```env
GROQ_API_KEY=your_groq_key_here
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
```

| Key | Where to get | Cost |
|-----|-------------|------|
| `GROQ_API_KEY` | https://console.groq.com | Free |
| `HUGGINGFACEHUB_API_TOKEN` | https://huggingface.co/settings/tokens | Free |

---

## 📥 Download Legal Documents

```bash
python collecting_data.py
```

This downloads:
- 📜 Indian Penal Code (IPC) 1860
- 📜 Constitution of India
- 📜 RTI Act 2005
- 📜 Consumer Protection Act 2019

If any download fails, the script will print the manual download link.

---

## 🗄️ Build Vector Store

Run **once** before starting the app:

```bash
python ingest.py
```

To rebuild from scratch (after adding new PDFs):
```bash
python ingest.py --reset
```

**Expected output:**
```
GPU  Checking GPU
   OK  NVIDIA GeForce RTX 5060  (8 GB)

DATA  Checking Data folder
   OK  constitution.pdf  (2029 KB)
   OK  ipc.pdf  (156 KB)
   ...

CHUNK  Splitting into chunks
   OK  1712 chunks  |  avg 824 chars

EMBED  Loading BAAI/bge-large-en-v1.5 on CUDA
   OK  Loaded in 3.2s  |  dimension=1024

BUILD  Embedding 1712 chunks
   OK  1712 vectors stored in 28.4s

TEST  Running test search
   OK  Hit 1: constitution.pdf  page 14
   ...

DONE! 1712 chunks stored in ./chroma_db
```

---

## 🚀 Run the App

```bash
streamlit run app.py
```

Opens at → http://localhost:8501

---

## 💬 Using the App

Once running:

1. Type any legal question in the chat box
2. Or click suggested questions in the sidebar
3. The answer shows with source document + page number
4. Ask follow-up questions — it remembers conversation history

**Example questions to try:**
- *"What is IPC Section 302?"*
- *"Who can file an RTI application?"*
- *"What are the fundamental rights?"*
- *"What are the conditions for using a government grant?"*
- *"What is the punishment for theft under IPC?"*

---

## 📓 Jupyter Notebook (For Learning)

Open `lexassist_rag.ipynb` in VS Code or JupyterLab to explore:

- Every RAG stage explained with markdown
- Code cells you can run step by step
- Advanced techniques: HyDE, Multi-Query, Contextual Compression
- RAGAS evaluation scaffold
- Architecture diagram at the end

```bash
pip install jupyter
jupyter notebook lexassist_rag.ipynb
```

---

## 🔧 Configuration

All settings are at the top of each file:

**`ingest.py`**
```python
EMBED_MODEL  = "BAAI/bge-large-en-v1.5"   # embedding model
CHUNK_SIZE   = 1000                         # chars per chunk
CHUNK_OVERLAP= 200                          # overlap between chunks
BATCH_SIZE   = 100                          # chunks per API call
```

**`app.py`**
```python
GROQ_MODEL   = "llama3-8b-8192"            # LLM model
RETRIEVER_K  = 4                            # chunks retrieved per query
```

---

## 🏗️ RAG Architecture

```
PDFs in Data/
     │
     ▼
DirectoryLoader → PyPDFLoader
     │
     ▼
RecursiveCharacterTextSplitter
(1000 chars, 200 overlap)
     │
     ▼
BAAI/bge-large-en-v1.5 Embeddings
(1024-dim vectors, runs on GPU)
     │
     ▼
ChromaDB Vector Store (persisted)
     │
     ▼ (at query time)
MMR Retriever (k=4, fetch_k=10)
     │
     ▼
ChatPromptTemplate
[system + chat_history + context + question]
     │
     ▼
Groq LLM (llama3-8b-8192)
     │
     ▼
Answer + Sources
```

---

## 📊 Evaluation (RAGAS)

The notebook includes a RAGAS evaluation scaffold measuring:

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Does the answer stick to retrieved context? |
| **Answer Relevancy** | Is the answer relevant to the question? |
| **Context Precision** | Was the most relevant context ranked first? |
| **Context Recall** | Was all important context retrieved? |

Uncomment Stage 8 in the notebook and add an OpenAI key to run full evaluation.

---

## 🐛 Common Issues

| Error | Fix |
|-------|-----|
| `0 chunks retrieved` | Run `python ingest.py --reset` to rebuild |
| `GROQ_API_KEY not found` | Check your `.env` file has the key |
| `GPU not detected` | Reinstall torch: `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| `ModuleNotFoundError` | Reinstall with pinned versions from Step 3 |
| `chroma_db locked` | Close all Streamlit/Jupyter processes, then delete `chroma_db/` |
| `504 timeout on embeddings` | Switch to local model — already handled in `ingest.py` |

---

## 🔭 What's Next (Roadmap)

- [ ] **Agents** — LLM decides whether to retrieve, calculate, or search
- [ ] **Tool Calling** — RAG + web search + calculator
- [ ] **LangGraph** — stateful multi-step workflows
- [ ] **Hybrid Search** — BM25 + semantic search combined
- [ ] **Cross-encoder Reranking** — improve retrieval precision

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain 0.3.7 |
| Vector DB | ChromaDB |
| Embeddings | BAAI/bge-large-en-v1.5 (HuggingFace) |
| LLM | Llama3-8b via Groq |
| UI | Streamlit |
| Evaluation | RAGAS |
| Language | Python 3.11 |

---

## 📄 Data Sources

All documents are freely available from official Indian government websites:

| Document | Source |
|----------|--------|
| Constitution of India | Ministry of Law & Justice |
| Indian Penal Code 1860 | IndiaCode.nic.in |
| RTI Act 2005 | IndiaCode.nic.in |
| Consumer Protection Act 2019 | ConsumerAffairs.nic.in |

---

## 🤝 Contributing

1. Fork the repo
2. Add new legal documents to `Data/`
3. Run `python ingest.py --reset`
4. Test with new queries
5. Submit a PR

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

*Built as a learning project while completing the GenAI with LangChain playlist — Campus X YouTube channel.*