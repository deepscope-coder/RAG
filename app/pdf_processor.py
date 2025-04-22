# app/pdf_processor.py
import re
from pathlib import Path
from typing import List, Dict
from PyPDF2 import PdfReader
from utils2 import logger

def pdf_to_chunks(pdf_path: Path, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
    if not pdf_path.exists() or not pdf_path.is_file():
        logger.error(f"❌ PDF file not found at {pdf_path}")
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
        chunks = []
        current_position = 0

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r'\s+', ' ', text).strip()
            words = text.split()

            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if len(chunk.split()) > 0:  # Skip empty chunks
                    chunk_id = f"{pdf_path.stem}_chunk_{len(chunks)}"
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk,
                        "page": page_num,
                        "source": pdf_path.name,
                        "start_pos": current_position
                    })
                    current_position += len(chunk.split())

        logger.info(f"📄 Processed {len(chunks)} chunks from {pdf_path.name}")
        return chunks
    except Exception as e:
        logger.error(f"❌ PDF processing failed for {pdf_path}: {e}")
        raise