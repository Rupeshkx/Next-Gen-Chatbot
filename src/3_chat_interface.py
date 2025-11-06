from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os

def init_chatbot():
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV")
    )
    index_name = "nextgen-chatbot-index"

    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index(index_name, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)

    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    return qa_chain

if __name__ == "__main__":
    chatbot = init_chatbot()
    chat_history = []

    print("🤖 Chatbot Ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        result = chatbot({"question": query, "chat_history": chat_history})
        print(f"Bot: {result['answer']}\n")
        chat_history.append((query, result["answer"]))
