import os
import torch
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any

class DenseRetriever:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None, batch_size: int = 128):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()

    def encode(self, texts: List[str], show_progress_bar: bool = True) -> np.ndarray:
        """
        Encode a list of texts into dense vectors and normalize them for inner-product retrieval.
        """
        # Encode with bf16 if cuda is available
        with torch.autocast(device_type=self.device if self.device != 'cpu' else 'cpu', enabled=(self.device == 'cuda')):
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True # BGE-M3 needs normalization for cosine similarity (equivalent to IP here)
            )
        return embeddings

class FaissIndexManager:
    def __init__(self, dim: int, index_type: str = "FlatIP"):
        self.dim = dim
        self.index_type = index_type
        
        if index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(dim)
        else:
            # We can expand to IVF/IVFPQ later for full scale, measuring recall/memory/throughput.
            raise ValueError(f"Unsupported index type {index_type} for now")
            
        # We need a separate IDMap since faiss indices default to sequential IDs
        # We'll use IndexIDMap to preserve our integer entity_ids if they fit in 64-bit, 
        # or we just keep a separate array mapping sequential Faiss ID to our entity_id string.
        # Given entity_ids can be strings (though they are integers in this dataset), let's keep an explicit array mapping.
        self.id_to_entity = []

    def add(self, embeddings: np.ndarray, entity_ids: List[str]):
        if len(embeddings) != len(entity_ids):
            raise ValueError("Embeddings and entity_ids length mismatch")
            
        self.index.add(embeddings.astype(np.float32))
        self.id_to_entity.extend(entity_ids)

    def search(self, query_embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Search the index. Returns (distances, faiss_indices, entity_ids)
        """
        distances, faiss_indices = self.index.search(query_embeddings.astype(np.float32), k)
        
        # Resolve entity IDs
        # faiss_indices has shape (num_queries, k)
        entity_ids = np.empty(faiss_indices.shape, dtype=object)
        for i in range(faiss_indices.shape[0]):
            for j in range(faiss_indices.shape[1]):
                idx = faiss_indices[i, j]
                if idx != -1:
                    entity_ids[i, j] = self.id_to_entity[idx]
                else:
                    entity_ids[i, j] = None
                    
        return distances, faiss_indices, entity_ids
        
    def save(self, filepath: str):
        faiss.write_index(self.index, filepath)
        # Also save the id map
        import pickle
        with open(filepath + ".idmap", 'wb') as f:
            pickle.dump(self.id_to_entity, f)
            
    def load(self, filepath: str):
        self.index = faiss.read_index(filepath)
        import pickle
        with open(filepath + ".idmap", 'rb') as f:
            self.id_to_entity = pickle.load(f)
