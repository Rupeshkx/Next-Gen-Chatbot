import os
from langchain.document_loaders import PyPDFLoader

def load_pdfs_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
    return documents

if __name__ == "__main__":
    data_path = "../data/reports"
    all_docs = load_pdfs_from_directory(data_path)
    print(f"✅ Loaded {len(all_docs)} documents.")
