# app/utils.py
import logging
from pathlib import Path
from typing import List, Dict

# --------------------- CONFIGURATION ---------------------
PINECONE_API_KEY = "pcsk_RQnDU_FtcV4diYkc79gq1JYq85HLv6EXyKx62exrXtivUy8zMEq6xEJwriJPzVzvCmmM9"  # Replace with your key
DEFAULT_PDF_DIR = Path(r"D:\arif\Docment_retravial\documents")  # Update this path
INDEX_NAME = "alg"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 32
CHUNK_SIZE = 400
OVERLAP = 50
ACCURACY_THRESHOLD = 0.3  # 30% threshold for match score
# ---------------------------------------------------------

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_config():
    return {
        "pinecone_api_key": PINECONE_API_KEY,
        "pdf_dir": DEFAULT_PDF_DIR,
        "index_name": INDEX_NAME,
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "accuracy_threshold": ACCURACY_THRESHOLD
    }