import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from retrieval.retriever import get_retriever  
from Reranker.reranker import bm25_rerank      
from langchain_core.documents import Document

# --- Helper Functions ---
def format_docs(docs: list[Document]) -> str:
    """Join document chunks into one context string."""
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(doc.page_content.strip() for doc in docs)

# --- Main Streamlit Application ---
def main():
    """
    Main function to run the Streamlit application.
    """
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("❌ Missing GROQ_API_KEY. Set it in your environment first.")
        st.stop()

    # --- App Configuration ---
    st.set_page_config(
        page_title="Resilience X HR Policy Assistant",
        page_icon="🤖",
        layout="centered"
    )
    st.title("🤖 Resilience X HR Policy Assistant")
    st.markdown("""
        Ask questions about our company HR policies.
        The assistant will only provide answers based on the official policy documents.
        Previous queries are remembered via LangChain memory.
    """)

    # --- Initialize Retriever and LLM ---
    # This assumes get_retriever is defined in your project to initialize your vector store
    try:
        retriever = get_retriever(index_type="hnsw", k=10)
    except Exception as e:
        st.error(f"❌ Failed to initialize the document retriever: {e}")
        st.stop()

    llm = ChatGroq(
        temperature=0,
        model_name="openai/gpt-oss-120b",
        api_key=groq_api_key,
    )

    # --- Initialize LangChain Memory ---
    # Using st.session_state to persist memory across reruns
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=False,
        )

    # --- Prompt Template ---
    # This template strictly instructs the LLM and includes conversation history
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an HR policy assistant for Resilience X.\n"
         "ONLY answer based on the provided context from documents.\n"
         "Do NOT use any knowledge outside the provided documents.\n"
         "If the answer is not in the documents, respond that you do not have enough information to answer.\n"
         "Keep answers concise and professional."),
        ("human",
         "Conversation History:\n{chat_history}\n\n"
         "Context from Documents:\n{context}\n\n"
         "User Query: {input}\n\n"
         "Final Answer:")
    ])

    # --- LLM Chain ---
    # The chain combines the prompt, LLM, and memory
    chat_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=st.session_state.memory,
        output_key="output",
        verbose=False,
    )
    
    # --- Display chat history ---
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # --- User Input and Response Logic ---
    if user_input := st.chat_input("Ask a question about our HR policies..."):
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(user_input)
        
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            with st.spinner("Searching official HR documents..."):
                # 1. Retrieve relevant documents
                retrieved_docs = retriever.get_relevant_documents(user_input)

                # 2. Rerank the results for better relevance
                reranked_docs = bm25_rerank(query=user_input, documents=retrieved_docs, top_n=5)

                # 3. Format documents into a single context string
                context = format_docs(reranked_docs)

            with st.spinner("Generating response..."):
                # 4. LLM reasons strictly based on retrieved context + LangChain memory
                response = chat_chain.predict(
                    input=user_input,
                    context=context
                ).strip()

            # Display assistant response in chat
            with st.chat_message("assistant"):
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Note: The LLMChain automatically handles saving the context to memory.
            # The manual `memory.save_context` call is not needed when using the chain's `predict` method.

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()