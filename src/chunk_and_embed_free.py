#  #  📄  loader_and_index.py
#  ---------------------------------
import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain Core
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ✅ IMPORTS for OpenRouter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.utils import get_from_env

# ------------------------------------------------------------------------------

# ✅ Load .env
load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
if not OPENROUTER_KEY:
    raise ValueError("❌ Missing OPENROUTER_KEY in environment variables.")

# ✅ OpenRouter Base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ✅ Models
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "deepseek/deepseek-chat"   # you can change to any OpenRouter model

# ✅ PDF path
PDF_PATH = Path("data/reports/sample.pdf")
if not PDF_PATH.exists():
    raise FileNotFoundError(f"❌ PDF not found: {PDF_PATH}")

# ------------------------------------------------------------------------------
# ✅ 1️⃣ LOAD PDF
print("[INFO] Loading PDF…")
loader = UnstructuredLoader(str(PDF_PATH))
documents = loader.load()
print(f"[✅] Loaded {len(documents)} document(s).")

# ✅ 2️⃣ SPLIT INTO CHUNKS
print("[INFO] Splitting into chunks…")
splitter = RecursiveCharacterTextSplitter(chunk_size=10_000, chunk_overlap=200)
docs = splitter.split_documents(documents)
print(f"[✅] Created {len(docs)} chunks.")

texts = [doc.page_content for doc in docs]

# ✅ 3️⃣ EMBEDDINGS — USING OPENROUTER
print(f"[INFO] Creating embeddings using OpenRouter model: {EMBEDDING_MODEL}")

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENROUTER_KEY,
    openai_api_base=OPENROUTER_BASE_URL
)

# ✅ Generate Embeddings
vectors = embeddings.embed_documents(texts)

for i, v in enumerate(vectors):
    print(f"Chunk {i} -> Embedding length: {len(v)} | First 10 dims: {v[:10]}")

# ✅ 4️⃣ STORE IN FAISS
print("[INFO] Building FAISS index…")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")
print("[✅] FAISS index saved at './faiss_index/'")


# ------------------------------------------------------------------------------

# # 5️⃣ Query helper (interactive demo)
# def answer(question: str, k: int = 3) -> str:
#     """
#     Return the best answer to *question* by:
#       • retrieving the top‑*k* passages via FAISS,
#       • passing them along with the question to the LLM.
#     """
#     # Retrieve
#     docs, _ = vectorstore.similarity_search_with_score(question, k=k)
#     context = "\n\n".join([f"<|doc{i+1}|>\n{d.page_content}" for i, d in enumerate(docs)])

#     # Build prompt (you can tweak this)
#     system_prompt = (
#         "You are a helpful assistant. "
#         "Answer the user’s question using only the provided documents."
#     )
#     user_prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

#     # Call the LLM
#     llm = ChatOpenAI(
#         openai_api_key=OPENAI_API_KEY,
#         model=LLM_MODEL,
#         temperature=0   # deterministic answers
#     )
#     msg = llm.invoke(
#         [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ]
#     )
#     return msg.content

# # Demo
# if __name__ == "__main__":
#     print("\n🏗️  FAISS index is ready. Try a quick query:")
#     query = input("Your question: ")
#     print("\n🔍  Searching…")
#     print(answer(query))