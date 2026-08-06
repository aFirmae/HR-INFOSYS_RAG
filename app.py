import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from retrieval.retriever import get_retriever  
from Reranker.reranker import bm25_rerank      
from langchain_core.documents import Document

def format_docs(docs: list[Document]) -> str:
    """Join document chunks into one context string."""
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(doc.page_content.strip() for doc in docs)

def main():
    """
    Main function to run the Streamlit application.
    """
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Missing GROQ_API_KEY. Set it in your environment first.")
        st.stop()

    st.set_page_config(
        page_title="JP Morgan HR Policy Assistant",
        page_icon="🤖",
        layout="centered"
    )
    st.title("🤖 JP Morgan HR Policy Assistant")
    st.markdown("""
        Ask questions about the company's HR transformation.
    """)

    try:
        retriever = get_retriever(index_type="hnsw", k=10)
    except Exception as e:
        st.error(f"Failed to initialize the document retriever: {e}")
        st.stop()

    llm = ChatGroq(
        temperature=0,
        model_name="openai/gpt-oss-120b",
        api_key=groq_api_key,
    )

    # Using st.session_state to persist memory across reruns
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=False,
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an HR policy assistant for JP Morgan.\n"
         "ONLY answer based on the provided context from documents.\n"
         "Do NOT use any knowledge outside the provided documents.\n"
         "If the answer is not in the documents, respond that you do not have enough information to answer.\n"
         "Keep answers concise, professional, and properly formatted using markdown (e.g., bullet points, bold text)."),
        ("human",
         "Conversation History:\n{chat_history}\n\n"
         "Context from Documents:\n{context}\n\n"
         "User Query: {input}\n\n"
         "Final Answer:")
    ])

    chat_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=st.session_state.memory,
        output_key="output",
        verbose=False,
    )
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(message["sources"]):
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page_number', 'N/A')
                        st.markdown(f"**Source {i+1}: {source} (Page {page})**")
                        st.markdown(f"> {doc.page_content.replace(chr(10), chr(10) + '> ')}")


    if user_input := st.chat_input("Ask a question about HR transformations..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            with st.spinner("Searching official HR documents..."):
                retrieved_docs = retriever.invoke(user_input)

                reranked_docs = bm25_rerank(query=user_input, documents=retrieved_docs, top_n=5)

                context = format_docs(reranked_docs)

            with st.spinner("Generating response..."):
                response = chat_chain.predict(
                    input=user_input,
                    context=context
                ).strip()

            with st.chat_message("assistant"):
                st.markdown(response)
                if reranked_docs:
                    with st.expander("📚 View Sources"):
                        for i, doc in enumerate(reranked_docs):
                            source = doc.metadata.get('source', 'Unknown')
                            page = doc.metadata.get('page_number', 'N/A')
                            st.markdown(f"**Source {i+1}: {source} (Page {page})**")
                            st.markdown(f"> {doc.page_content.replace(chr(10), chr(10) + '> ')}")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": reranked_docs
            })

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()