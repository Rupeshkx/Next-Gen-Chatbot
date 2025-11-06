# Next-Gen-Chatbot
Goal:
    Read huge PDF data from a shared path
    Split text into chunks using LangChain
    Store embeddings in Pinecone
    Query contextually via OpenAI GPT
    Serve via FastAPI and optionally Streamlit UI
Folder Structure:
    nextgen-chatbot/
    ├── .env                 # API keys (dummy placeholders)
    ├── requirements.txt     # Python dependencies
    ├── config/
    │   └── config.yaml      # Config: data path, chunk size, index
    ├── src/
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── chunk_and_embed.py
    │   ├── chat_service.py
    │   ├── api_server.py
    │   └── ui_app.py
    ├── Dockerfile
    ├── render.yaml           # Render deployment
    ├── run.sh
    └── README.md
