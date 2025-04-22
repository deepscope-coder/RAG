# app/main.py
import argparse
import sys
from pathlib import Path
from typing import List
from pinecone import Pinecone
from concurrent.futures import ThreadPoolExecutor
from utils2 import get_config, logger  # Updated to utils2
from pinecone_utils import initialize_pinecone, create_or_connect_index, upsert_vectors
from pdf_processor import pdf_to_chunks
from embedding import generate_embeddings, process_batch
from search_interface import search_pdf, index_pdfs, load_hf_model, main as streamlit_main

def run_cli(pdf_paths: List[Path], pc: Pinecone, tokenizer, model, query: str = None):
    """Run the command-line interface for indexing and searching."""
    index = index_pdfs(pdf_paths, pc, tokenizer, model)
    
    if query:
        results = search_pdf(index, query, tokenizer, model)
        if results:
            for match in results:
                print(f"\nMatch (Score: {match.score:.3f}, Source: {match.metadata['source']}, Page: {match.metadata['page']}):")
                print(f"{match.metadata['text'][:200]}...")
    else:
        while True:
            query = input("Enter query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            results = search_pdf(index, query, tokenizer, model)
            if results:
                for match in results:
                    print(f"\nMatch (Score: {match.score:.3f}, Source: {match.metadata['source']}, Page: {match.metadata['page']}):")
                    print(f"{match.metadata['text'][:200]}...")

def main():
    """Main function with CLI support for multiple PDFs and optional Streamlit interface."""
    parser = argparse.ArgumentParser(description="Document Retrieval System")
    parser.add_argument("--pdf-dir", type=Path, default=get_config()["pdf_dir"], help="Directory containing PDF files")
    parser.add_argument("--query", type=str, help="Single search query")
    parser.add_argument("--streamlit", action="store_true", help="Launch the Streamlit web interface")
    args = parser.parse_args()

    if args.streamlit:
        streamlit_main()  # Call the Streamlit main function directly
    else:
        try:
            # Find all PDF files in the directory
            pdf_paths = list(args.pdf_dir.glob("*.pdf"))
            if not pdf_paths:
                logger.error(f"❌ No PDF files found in {args.pdf_dir}")
                sys.exit(1)
            logger.info(f"ℹ️ Found {len(pdf_paths)} PDF files: {[p.name for p in pdf_paths]}")

            pc = initialize_pinecone(get_config()["pinecone_api_key"])
            tokenizer, model = load_hf_model(get_config()["model_name"])
            run_cli(pdf_paths, pc, tokenizer, model, args.query)

        except Exception as e:
            logger.error(f"❌ Program failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()