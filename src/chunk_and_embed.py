'''Update chunk_and_embed.py for FAISS instead of pinecone database
Replace Pinecone (Free Option)

Since Pinecone free tier may require signup and you don’t want to pay:

We can use FAISS, a free local vector database.

This works without signup or payment.

pip install faiss-cpu

'''
# src/chunk_and_embed.py

# src/chunk_and_embed.py

# import os
# from dotenv import load_dotenv

# # LangChain imports
# #from langchain.document_loaders.unstructured import UnstructuredPDFLoader
# from langchain_unstructured import UnstructuredLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# #from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
# #from langchain.vectorstores import FAISS
# from langchain_community.vectorstores import FAISS

# # Load environment variables from .env
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# if not OPENAI_API_KEY:
#     raise ValueError("Please set your OPENAI_API_KEY in the .env file.")

# # Path to your PDF(s)
# PDF_PATH = "data/reports/sample.pdf"  # <- replace with your PDF file

# # Step 1: Load the PDF
# print("[INFO] Loading PDF...")
# loader = UnstructuredLoader(PDF_PATH)
# documents = loader.load()
# print(f"[INFO] Loaded {len(documents)} document(s).")

# # Step 2: Split documents into smaller chunks
# print("[INFO] Splitting documents into chunks...")
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=10000,  # characters per chunk
#     chunk_overlap=1000  # overlap characters
# )
# docs = text_splitter.split_documents(documents)
# print(f"[INFO] Created {len(docs)} chunks.")

# # Step 3: Create embeddings
# print("[INFO] Creating embeddings...")
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# #print("embeddings", embeddings)


# # texts = [doc.page_content for doc in docs]
# # print(texts)
# # # Generate embeddings
# # vectors = embeddings.embed_documents(texts)

# # for i, v in enumerate(vectors):
# #     print(f"Chunk {i} embedding length: {len(v)}")
# #     print(f"First 10 dimensions: {v[:10]}")



# # Step 4: Store in FAISS vector store (local)
# print("[INFO] Storing embeddings in FAISS...")
# vectorstore = FAISS.from_documents(docs, embeddings)
# vectorstore.save_local("faiss_index")
# print("[INFO] FAISS index saved to 'faiss_index' folder.")

# print("[SUCCESS] Done! You can now query your documents using FAISS.")




# src/chunk_and_embed.py

import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Choose embedding provider via ENV variable: "OPENAI" or "DEEPSEEK"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "OPENAI").upper()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if EMBEDDING_PROVIDER == "OPENAI" and not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in .env for OpenAI embeddings.")
if EMBEDDING_PROVIDER == "DEEPSEEK" and not DEEPSEEK_API_KEY:
    raise ValueError("Please set your DEEPSEEK_API_KEY in .env for DeepSeek embeddings.")

# Path to PDF(s)
PDF_PATH = "data/reports/sample.pdf"

# Step 1: Load PDF
print("[INFO] Loading PDF...")
loader = UnstructuredLoader(PDF_PATH)
documents = loader.load()
print(f"[INFO] Loaded {len(documents)} document(s).")

# Step 2: Split documents
print("[INFO] Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100000,
    chunk_overlap=10
)
docs = text_splitter.split_documents(documents)
print(f"[INFO] Created {len(docs)} chunks.")
texts = [doc.page_content for doc in docs]
print(f"[INFO] Created docs parts as text :{texts}" )

# Step 3: Create embeddings
print(f"[INFO] Creating embeddings using {EMBEDDING_PROVIDER}...")

if EMBEDDING_PROVIDER == "OPENAI":
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

elif EMBEDDING_PROVIDER == "DEEPSEEK":
    import requests

    class DeepSeekEmbeddings:
        def __init__(self, api_key: str, model: str = "deepseek-embedding-v2"):
            self.api_key = api_key
            self.model = model
            self.endpoint = "https://api.deepseek.ai/v1/embeddings"

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "input": texts
            }
            response = requests.post(self.endpoint, json=payload, headers=headers)
            if response.status_code != 200:
                raise ValueError(f"DeepSeek API error: {response.text}")
            embeddings = response.json()["data"]
            return [item["embedding"] for item in embeddings]

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

    embeddings = DeepSeekEmbeddings(api_key=DEEPSEEK_API_KEY)

else:
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")



# Step 4: Store in FAISS vector store (local)
print("[INFO] Storing embeddings in FAISS...")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")
print("[INFO] FAISS index saved to 'faiss_index' folder.")

print("[SUCCESS] Done! You can now query your documents using FAISS.")

