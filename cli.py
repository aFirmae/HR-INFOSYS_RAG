import argparse
import logging
import os
import warnings

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from retrieval.retriever import get_retriever
from Reranker.reranker import bm25_rerank


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(doc.page_content.strip() for doc in docs)


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an HR policy assistant for Resilience X.\n"
     "ONLY answer based on the provided context from documents.\n"
     "Do NOT use any knowledge outside the provided documents.\n"
     "If the answer is not in the documents, respond that you do not have enough information to answer.\n"
     "Keep answers concise and professional."),
    ("human",
     "Context from Documents:\n{context}\n\n"
     "User Query: {input}\n\n"
     "Final Answer:")
])


def answer(query: str) -> str:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise SystemExit("Missing GROQ_API_KEY. Set it in your .env file first.")

    retriever = get_retriever(index_type="hnsw", k=10)
    retrieved_docs = retriever.invoke(query)
    reranked_docs = bm25_rerank(query=query, documents=retrieved_docs, top_n=5)
    context = format_docs(reranked_docs)

    llm = ChatGroq(temperature=0, model_name="openai/gpt-oss-120b", api_key=groq_api_key)
    chain = PROMPT | llm
    response = chain.invoke({"context": context, "input": query})
    return response.content.strip()


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Query the HR Policy Q&A Assistant from the command line.")
    parser.add_argument("--query", required=True, help="Question to ask about HR policies.")
    args = parser.parse_args()

    print(answer(args.query))


if __name__ == "__main__":
    main()
