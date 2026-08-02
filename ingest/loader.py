import os
import logging
from typing import Generator, List
from pypdf import PdfReader
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFStreamingLoader:
    """
    Streams individual pages from all PDFs in a folder using PdfReader (more robust than PyPDFLoader).
    Adds rich metadata to each page.
    """
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def stream_documents(self) -> Generator[Document, None, None]:
        """Streams documents page by page with metadata."""
        if not os.path.exists(self.folder_path):
            logger.error(f"Folder not found: {self.folder_path}")
            return

        for filename in os.listdir(self.folder_path):
            if not filename.lower().endswith(".pdf"):
                continue

            full_path = os.path.join(self.folder_path, filename)
            logger.info(f"📄 Processing: {filename}")

            try:
                reader = PdfReader(full_path)
            except Exception as e:
                logger.error(f"❌ Failed to open PDF {filename}: {e}")
                continue

            if not reader.pages:
                logger.warning(f"⚠️ No pages found in PDF: {filename}")
                continue

            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                except Exception as e:
                    logger.warning(f"⚠️ Could not extract text from page {i+1} of {filename}: {e}")
                    continue

                if not text or not text.strip():
                    continue

                doc = Document(
                    page_content=text.strip(),
                    metadata={
                        "source": filename,
                        "file_path": full_path,
                        "page_number": i + 1,
                        "type": "hr_policy",
                    },
                )
                yield doc

    def load_all(self) -> List[Document]:
        """Eagerly loads all documents into memory (use with caution for very large PDFs)."""
        return list(self.stream_documents())
