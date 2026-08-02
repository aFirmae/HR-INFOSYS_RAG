# bm25_retriever.py
import os
import re
from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens

def load_pdfs_from_folder(folder_path: str) -> List[Document]:
    """Load all PDFs in folder and return as LangChain Document chunks"""
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # Split into chunks
            chunks = splitter.split_documents(docs)
            # Add metadata: file name + unique id
            for i, chunk in enumerate(chunks):
                chunk.metadata["source"] = filename
                chunk.metadata["id"] = f"{filename}::chunk_{i}"
            all_docs.extend(chunks)

    return all_docs

class BM25Retriever:
    def __init__(self, docs: List[Document], tokenizer=simple_tokenize):
        self.docs = docs
        self.ids = [doc.metadata.get("id", str(i)) for i, doc in enumerate(docs)]
        self.tokenizer = tokenizer
        self.tokenized_corpus = [self.tokenizer(d.page_content) for d in docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_relevant_documents(self, query: str, k: int = 10):
        q_tokens = self.tokenizer(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in ranked_idx:
            d = self.docs[idx]
            md = dict(d.metadata or {})
            md["bm25_score"] = float(scores[idx])
            results.append(Document(page_content=d.page_content, metadata=md))
        return results


# =====================
# Usage example
# =====================
if __name__ == "__main__":
    folder_path = r"D:\AGENTIC_AI\PROJECTS\hr-assistant\data"
    all_docs = load_pdfs_from_folder(folder_path)
    print(f"Loaded {len(all_docs)} chunks from PDFs.")

    bm25_retriever = BM25Retriever(all_docs)

    query = "When do I get paid each month?"
    results = bm25_retriever.get_relevant_documents(query, k=10)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"BM25 Score: {doc.metadata.get('bm25_score')}")
        print(doc.page_content[:300], "...\n")
