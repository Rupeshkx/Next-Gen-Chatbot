# 📄 chunk_and_embed.py
import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ✅ OpenRouter Embeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ✅ OpenRouter Credentials
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

if not OPENROUTER_KEY:
    raise ValueError("❌ Missing OPENROUTER_KEY in environment variables!")

# ✅ PDF Path
PDF_PATH = Path("data/reports/sample.pdf")


def chunk_and_embed():
    """Load PDF → Split → Embed → Build FAISS & return vectorstore"""

    # ✅ 1. Load PDF
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"❌ PDF not found: {PDF_PATH}")

    print("[INFO] Loading PDF...")
    loader = UnstructuredLoader(str(PDF_PATH))
    documents = loader.load()
    print(f"[✅] Loaded {len(documents)} documents")

    # ✅ 2. Split into chunks
    print("[INFO] Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    texts = [doc.page_content for doc in chunks]
    print(f"[✅] Created {len(chunks)} chunks")

    # ✅ 3. Create Embeddings (OpenRouter)
    print("[INFO] Creating embeddings using text-embedding-3-large...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENROUTER_KEY,
        openai_api_base=OPENROUTER_BASE
    )

    # ✅ Generate Embeddings
    vectors = embeddings.embed_documents(texts)

    for i, v in enumerate(vectors):
        print(f"Chunk {i} -> Embedding length: {len(v)} | First 10 dims: {v[:10]}")
    # ✅ 4. Build FAISS index
    print("[INFO] Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("[✅] FAISS index built successfully!")

    return vectorstore

#data = chunk_and_embed()