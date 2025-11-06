import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel

from chunk_and_embed_final import chunk_and_embed

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

app = FastAPI()

# ✅ Load vectorstore from chunk_and_embed()
vectorstore = chunk_and_embed()
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ✅ Use DeepSeek Chat LLM from OpenRouter
llm = ChatOpenAI(
    model="deepseek/deepseek-chat",
    temperature=0,
    openai_api_key=OPENROUTER_KEY,
    openai_api_base=OPENROUTER_BASE
)

chatbot = ConversationalRetrievalChain.from_llm(llm, retriever)


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask(query: Query):
    result = chatbot({"question": query.question, "chat_history": []})
    return {"answer": result["answer"]}
