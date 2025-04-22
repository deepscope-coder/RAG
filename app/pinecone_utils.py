# app/pinecone_utils.py
from tenacity import retry, stop_after_attempt, wait_exponential
from pinecone import Pinecone, ServerlessSpec
from utils2 import logger
import time

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def initialize_pinecone(api_key: str) -> Pinecone:
    try:
        pc = Pinecone(api_key=api_key)
        logger.info("✅ Pinecone initialized successfully")
        return pc
    except Exception as e:
        logger.error(f"❌ Pinecone initialization failed: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upsert_vectors(index, vectors, namespace: str):
    return index.upsert(vectors=vectors, namespace=namespace)

def create_or_connect_index(pc: Pinecone, index_name: str) -> Pinecone.Index:
    try:
        existing_indexes = pc.list_indexes().names()
        if index_name not in existing_indexes:
            logger.info(f"ℹ️ Index '{index_name}' does not exist. Creating it...")
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"⏳ Waiting 30 seconds for index '{index_name}' to initialize...")
            time.sleep(30)
        else:
            logger.info(f"ℹ️ Index '{index_name}' already exists. Connecting to it...")
        
        index = pc.Index(index_name)
        return index
    except Exception as e:
        logger.error(f"❌ Failed to create or connect to index '{index_name}': {e}")
        raise