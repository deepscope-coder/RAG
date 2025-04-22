# app/embedding.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict  # Updated to include Dict
from utils2 import logger

def generate_embeddings(texts: List[str], tokenizer, model) -> List[List[float]]:
    try:
        encoded_input = tokenizer(
            texts, padding=True, truncation=True, max_length=384, return_tensors='pt'
        )
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        token_embeddings = model_output.last_hidden_state
        input_mask = encoded_input['attention_mask']
        input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(embeddings, p=2, dim=1).tolist()
    
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        raise

def process_batch(batch: List[Dict], tokenizer, model) -> List[Dict]:
    """Process a batch of chunks into vectors with embeddings."""
    texts = [item["text"] for item in batch]
    embeddings = generate_embeddings(texts, tokenizer, model)
    return [{
        "id": item["id"],
        "values": emb,
        "metadata": {
            "text": item["text"],
            "source": item["source"],
            "page": item["page"]
        }
    } for item, emb in zip(batch, embeddings)]