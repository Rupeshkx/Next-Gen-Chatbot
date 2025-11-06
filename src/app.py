import streamlit as st
from dotenv import load_dotenv
import os

from chunk_and_embed_final import chunk_and_embed
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

if not OPENROUTER_KEY:
    st.error("❌ Missing OPENROUTER_KEY in .env")
    st.stop()

# -----------------------------
# ✅ Streamlit Page Settings
# -----------------------------
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PDF Chatbot")
st.write("Ask questions to your PDF in real time!")

# -----------------------------
# ✅ Load Vectorstore (cached)
# -----------------------------
@st.cache_resource
def load_vectorstore():
    return chunk_and_embed()

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# -----------------------------
# ✅ LLM (DeepSeek via OpenRouter)
# -----------------------------
llm = ChatOpenAI(
    model="deepseek/deepseek-chat",
    temperature=0,
    openai_api_key=OPENROUTER_KEY,
    openai_api_base=OPENROUTER_BASE,
)

chatbot = ConversationalRetrievalChain.from_llm(llm, retriever)

# -----------------------------
# ✅ Initialize Chat History
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# ✅ User Input
# -----------------------------
user_query = st.text_input("Ask a question about the PDF:", "")

if user_query:
    with st.spinner("Thinking..."):
        response = chatbot.invoke({
            "question": user_query,
            "chat_history": st.session_state.chat_history,
        })

    answer = response["answer"]

    # Update chat history
    st.session_state.chat_history.append((user_query, answer))

# -----------------------------
# ✅ Display Chat History
# -----------------------------
st.subheader("💬 Conversation")

for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {q}")
    st.markdown(f"**🤖 Bot:** {a}")
    st.markdown("---")
