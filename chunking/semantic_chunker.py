from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],  
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Splits input documents into semantic chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self.splitter.split_documents([doc])
            for chunk in chunks:
                # Preserve original metadata
                chunk.metadata.update(doc.metadata)
            all_chunks.extend(chunks)
        return all_chunks