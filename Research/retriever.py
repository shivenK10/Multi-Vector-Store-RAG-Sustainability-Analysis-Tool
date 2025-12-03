"""
Retriever module for sustainability framework vector stores.
Handles context retrieval from multiple FAISS indices with metadata.
"""
import os
import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer


class SustainabilityRetriever:
    """Retrieves relevant context from sustainability framework vector stores."""
    
    def __init__(self, vector_stores_dir: str = "vector_stores"):
        """
        Initialize retriever with FAISS vector stores.
        
        Args:
            vector_stores_dir: Directory containing FAISS indices and metadata
        """
        self.vector_stores_dir = Path(vector_stores_dir)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.stores = {}
        self.metadata = {}
        self._load_all_stores()
    
    def _load_all_stores(self):
        """Load all FAISS vector stores and their metadata."""
        store_files = {
            "un_2030_agenda": "un_2030_agenda",
            "sdgs": "sdgs",
            "eu_taxonomy": "eu_taxonomy_user_guide",
            "csrd": "csrd_fact_sheet"
        }
        
        for store_name, file_prefix in store_files.items():
            index_path = self.vector_stores_dir / f"{file_prefix}_faiss.index"
            metadata_path = self.vector_stores_dir / f"{file_prefix}_metadata.pkl"
            
            if index_path.exists() and metadata_path.exists():
                self.stores[store_name] = faiss.read_index(str(index_path))
                with open(metadata_path, 'rb') as f:
                    self.metadata[store_name] = pickle.load(f)
                print(f"Loaded {store_name} vector store")
            else:
                print(f"Warning: Missing vector store files for {store_name}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 25,
        framework: Optional[str] = None,
        min_score: float = 0.15
    ) -> List[Dict]:
        """
        Retrieve relevant context chunks for a query.
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve per framework
            framework: Specific framework to search (None = all)
            min_score: Minimum similarity score threshold
        
        Returns:
            List of dicts with 'text', 'source', 'score', 'metadata'
        """
        # Encode and normalize query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        all_results = []
        
        # Search specified framework or all
        frameworks_to_search = [framework] if framework else self.stores.keys()
        
        for store_name in frameworks_to_search:
            if store_name not in self.stores:
                continue
            
            # Retrieve more candidates for better coverage
            scores, indices = self.stores[store_name].search(
                query_embedding.astype(np.float32),
                top_k * 2
            )
            
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score > min_score:
                    metadata = self.metadata[store_name][idx]
                    all_results.append({
                        'text': metadata['text'],
                        'source': store_name,
                        'score': float(score),
                        'metadata': metadata['metadata']
                    })
        
        # Sort by score and limit
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k * 2]
    
    def get_framework_stats(self) -> Dict[str, int]:
        """Get chunk counts for each loaded framework."""
        return {
            name: len(meta)
            for name, meta in self.metadata.items()
        }


if __name__ == "__main__":
    # Test retriever
    retriever = SustainabilityRetriever()
    
    print("\nFramework Statistics:")
    for framework, count in retriever.get_framework_stats().items():
        print(f"  {framework}: {count} chunks")
    
    test_query = "How can renewable energy contribute to climate goals?"
    print(f"\nTest Query: {test_query}\n")
    
    results = retriever.retrieve(test_query, top_k=5)
    print(f"Retrieved {len(results)} chunks:\n")
    
    for i, result in enumerate(results[:3], 1):
        print(f"--- Result {i} (Score: {result['score']:.4f}) ---")
        print(f"Source: {result['source']}")
        print(f"Text: {result['text'][:150]}...\n")
