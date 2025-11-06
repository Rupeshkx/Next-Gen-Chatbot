# 📄 chat.py

from chunk_and_embed_final import chunk_and_embed
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

if not OPENROUTER_KEY:
    raise ValueError("❌ Missing OPENROUTER_KEY in .env")

# ✅ 1. Load the vectorstore (FAISS)
vectorstore = chunk_and_embed()

# ✅ 2. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ✅ 3. Create LLM (DeepSeek on OpenRouter)
llm = ChatOpenAI(
    model="deepseek/deepseek-chat",  # you can change to any OpenRouter model
    temperature=0,
    openai_api_key=OPENROUTER_KEY,
    openai_api_base=OPENROUTER_BASE,
)

# ✅ 4. Create conversational chain
chatbot = ConversationalRetrievalChain.from_llm(llm, retriever)

# ✅ 5. Ask a test question
query = "What is in the PDF?"

result = chatbot.invoke({"question": query, "chat_history": []})

print("\n📌 ANSWER:")
print(result["answer"])
