# app/search_interface.py
import streamlit as st
from pathlib import Path
from typing import List
from transformers import AutoTokenizer, AutoModel
from utils2 import logger, get_config
from pinecone_utils import initialize_pinecone, create_or_connect_index
from pdf_processor import pdf_to_chunks
from embedding import generate_embeddings, process_batch
from concurrent.futures import ThreadPoolExecutor
from pinecone import Pinecone

def load_hf_model(model_name: str) -> tuple:
    """Load Hugging Face tokenizer and model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        logger.info("✅ Hugging Face model loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        raise

def index_pdfs(pdf_paths: List[Path], pc: Pinecone, tokenizer, model, batch_size: int = 32):
    """Index multiple PDFs into Pinecone with progress message."""
    index = create_or_connect_index(pc, get_config()["index_name"])
    
    total_vectors = 0
    status_text = st.empty()  # Placeholder for status updates

    for pdf_path in pdf_paths:
        data = pdf_to_chunks(pdf_path)
        status_text.text(f"Processing {pdf_path.name}...")  # Update status for current file
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
            vector_batches = list(executor.map(lambda b: process_batch(b, tokenizer, model), batches))
            
            for i, batch in enumerate(vector_batches):
                try:
                    index.upsert(vectors=batch, namespace="pdf-ns")
                    total_vectors += len(batch)
                    status_text.text(f"Indexing {pdf_path.name} - Batch {i + 1}/{len(batches)}")  # Update status for batch
                except Exception as e:
                    logger.warning(f"⚠️ Batch {i + 1} for {pdf_path.name} failed: {e}")
                    status_text.text(f"Error processing batch {i + 1} for {pdf_path.name}")

    logger.info(f"✅ Successfully upserted {total_vectors} vectors across all PDFs")
    status_text.text("Indexing completed successfully!")  # Final status update
    return index

def search_pdf(index, query: str, tokenizer, model, top_k: int = 3) -> List:
    """Search the indexed PDFs for a query with source, page numbers, and accuracy threshold."""
    try:
        query_embed = generate_embeddings([query], tokenizer, model)[0]
        logger.info(f"🔍 Searching for: '{query}'")
        
        results = index.query(
            vector=query_embed,
            top_k=top_k,
            include_metadata=True,
            namespace="pdf-ns"
        )
        
        if not results.matches:
            logger.info("❌ No matches found.")
            return []
        
        top_score = results.matches[0].score
        if top_score < get_config()["accuracy_threshold"]:
            logger.info(f"❌ Top match score ({top_score:.3f}) below 30%. No confident answer available.")
            st.warning("Sorry, we don't know.")
            return []

        logger.info(f"✅ Found {len(results.matches)} matches:")
        return results.matches
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        return []

def main():
    st.title("Document Retrieval System")
    st.write("Search through your indexed PDF documents using natural language queries.")

    # Initialize Pinecone and load model
    pc = initialize_pinecone(get_config()["pinecone_api_key"])
    tokenizer, model = load_hf_model(get_config()["model_name"])
    index = create_or_connect_index(pc, get_config()["index_name"])

    # Option to upload new PDFs
    st.subheader("Index New PDFs")
    uploaded_files = st.file_uploader("Upload PDF files to index", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Index PDFs"):
        pdf_paths = [Path("temp_" + str(i) + ".pdf") for i in range(len(uploaded_files))]
        for uploaded_file, pdf_path in zip(uploaded_files, pdf_paths):
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        index_pdfs(pdf_paths, pc, tokenizer, model)
        for pdf_path in pdf_paths:
            pdf_path.unlink()  # Clean up temporary files
        st.success("PDFs indexed successfully!")

    # Search interface
    st.subheader("Search Documents")
    query = st.text_input("Enter your search query:")
    if query and st.button("Search"):
        results = search_pdf(index, query, tokenizer, model)
        if results:
            st.write(f"Found {len(results)} matches:")
            for match in results:
                st.write(f"**Score:** {match.score:.3f}")
                st.write(f"**Source:** {match.metadata['source']}")
                st.write(f"**Page:** {match.metadata['page']}")
                st.write(f"**Text:** {match.metadata['text'][:200]}...")
                st.write("---")

if __name__ == "__main__":
    main()