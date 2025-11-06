from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import os

def chunk_and_embed(docs):
    # 1️⃣ Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)

    print(f"✅ Created {len(texts)} text chunks.")

    # 2️⃣ Initialize Pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV")
    )

    index_name = "nextgen-chatbot-index"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=1536)

    # 3️⃣ Create embeddings and store in Pinecone
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_documents(texts, embeddings, index_name=index_name)
    print("✅ Data stored successfully in Pinecone.")

if __name__ == "__main__":
    from data_ingestion import load_pdfs_from_directory
    docs = load_pdfs_from_directory("../data/reports")
    chunk_and_embed(docs)
