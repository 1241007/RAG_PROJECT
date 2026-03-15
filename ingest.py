"""
LexAssist — Document Ingestion Pipeline

Builds a Chroma vector database from local documents.

Usage
-----

Build database
    python ingest.py

Reset database
    python ingest.py --reset

Custom data directory
    python ingest.py --data ./documents
"""

import os
import sys
import time
import argparse
import shutil
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# =============================
# CONFIG
# =============================

DATA_DIR = "Data"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "lexassist"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 64

EMBED_MODEL = "BAAI/bge-large-en-v1.5"

SUPPORTED_TYPES = {
    "*.pdf": PyPDFLoader,
    "*.txt": TextLoader,
    "*.md": TextLoader,
    "*.docx": UnstructuredWordDocumentLoader,
}

# =============================
# LOGGING
# =============================

def log(title):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def ok(msg):
    print(f" OK  {msg}")

def info(msg):
    print(f" >>  {msg}")

def warn(msg):
    print(f" WARN {msg}")

def err(msg):
    print(f" ERR {msg}")


# =============================
# UTILITIES
# =============================

def file_hash(path):
    """Create SHA1 hash of file"""
    sha = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def detect_device():
    """Detect GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory // 1024**3
            ok(f"GPU detected: {gpu} ({mem} GB)")
            return "cuda"
    except Exception:
        pass

    warn("GPU not detected — using CPU")
    return "cpu"


# =============================
# DOCUMENT LOADING
# =============================

def load_documents(data_dir):

    log("Loading documents")

    all_docs = []

    for pattern, loader in SUPPORTED_TYPES.items():

        loader_instance = DirectoryLoader(
            data_dir,
            glob=pattern,
            loader_cls=loader,
            show_progress=True,
            silent_errors=True
        )

        docs = loader_instance.load()

        for d in docs:
            path = Path(d.metadata["source"])
            d.metadata.update({
                "file_name": path.name,
                "file_path": str(path),
                "file_hash": file_hash(path),
            })

        all_docs.extend(docs)

    ok(f"{len(all_docs)} documents loaded")

    if not all_docs:
        err("No supported documents found")
        sys.exit(1)

    return all_docs


# =============================
# CHUNKING
# =============================

def chunk_documents(docs):

    log("Chunking documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    sizes = [len(c.page_content) for c in chunks]

    ok(f"{len(chunks)} chunks created")
    info(f"Average size: {sum(sizes)//len(sizes)} chars")

    return chunks


# =============================
# VECTORSTORE BUILD
# =============================

def build_vectorstore(chunks, embeddings):

    log("Building vector database")

    start = time.time()

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    total = (len(chunks) - 1) // BATCH_SIZE + 1

    for i in range(0, len(chunks), BATCH_SIZE):

        batch = chunks[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        info(f"Batch {batch_num}/{total}  ({len(batch)} chunks)")

        vs.add_documents(batch)

    vs.persist()

    count = vs._collection.count()

    ok(f"{count} vectors stored")
    info(f"Time: {time.time()-start:.2f}s")

    return vs


# =============================
# TEST QUERY
# =============================

def test_search(vs):

    log("Testing vector search")

    results = vs.similarity_search("fundamental rights", k=3)

    for i, doc in enumerate(results, 1):

        print("\n--------------------------------")
        print(f"Result {i}")
        print(f"File: {doc.metadata.get('file_name')}")
        print(f"Page: {doc.metadata.get('page','?')}")
        print(doc.page_content[:200].replace("\n"," "))


# =============================
# MAIN INGEST FUNCTION
# =============================

def ingest(reset=False, data_dir=DATA_DIR):

    device = detect_device()

    if reset and Path(CHROMA_DIR).exists():

        log("Resetting vector database")
        shutil.rmtree(CHROMA_DIR)
        ok("Old database removed")

    docs = load_documents(data_dir)

    chunks = chunk_documents(docs)

    log("Loading embedding model")

    t0 = time.time()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )

    ok(f"Model loaded in {time.time()-t0:.2f}s")

    vs = build_vectorstore(chunks, embeddings)

    test_search(vs)

    print("\n")
    print("="*60)
    print(" INGESTION COMPLETE")
    print("="*60)
    print(f" Vector DB : {CHROMA_DIR}")
    print(f" Model     : {EMBED_MODEL}")
    print(f" Device    : {device}")
    print("\nNext step:")
    print("   streamlit run app.py")
    print("="*60)


# =============================
# CLI
# =============================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing vector DB"
    )

    parser.add_argument(
        "--data",
        type=str,
        default=DATA_DIR,
        help="Document folder"
    )

    args = parser.parse_args()

    ingest(reset=args.reset, data_dir=args.data)