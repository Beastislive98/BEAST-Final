import faiss
import numpy as np
import logging
from typing import Dict, Any, Optional, List
import time
import os
import json

class PatternMemoryBank:
    def __init__(self, dimension: int = 4096, max_patterns: int = 1000000):
        self.dimension = dimension
        self.max_patterns = max_patterns
        self.memory: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
        
        # Create enhanced FAISS index with better configuration
        # Use L2 distance for pattern similarity
        self.index = faiss.IndexFlatL2(dimension)
        
        # Attempt to use GPU if available
        try:
            if faiss.get_num_gpus() > 0:
                logging.info(f"FAISS: GPU support detected! Using GPU acceleration")
                self.gpu_resources = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                self.index = gpu_index
                self.using_gpu = True
            else:
                self.using_gpu = False
        except Exception as e:
            logging.warning(f"FAISS GPU acceleration failed: {e}, using CPU backend")
            self.using_gpu = False
            
        # Track memory usage
        self.total_memory_mb = 0
        self.last_prune_time = time.time()
        
        # Load existing patterns if available
        self._load_patterns()
        
        logging.info(f"PatternMemoryBank initialized with dimension={dimension}, max_patterns={max_patterns}, using_gpu={self.using_gpu}")

    def add_pattern(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> int:
        """
        Add a pattern to the memory bank with optimized memory management
        """
        if embedding.shape != (self.dimension,):
            logging.error(f"Embedding dimension mismatch: got {embedding.shape}, expected ({self.dimension},)")
            return -1
            
        # Ensure embedding is the correct data type for FAISS
        embedding = embedding.astype(np.float32)
        
        # Check if we need to prune (if we hit max capacity)
        if self.index.ntotal >= self.max_patterns:
            self._prune_old_patterns()

        # Add pattern to FAISS index
        self.index.add(np.expand_dims(embedding, axis=0))
        
        # Save embedding in metadata for potential rebuilding
        pattern_metadata = metadata.copy()
        
        # Store a compressed version of the embedding (save memory)
        # We don't need the full precision for rebuilding
        compressed_emb = embedding.astype(np.float16)
        pattern_metadata["embedding"] = compressed_emb
        
        # Track memory usage (rough estimate)
        pattern_size_mb = (embedding.nbytes + 1024) / (1024 * 1024)  # Add 1KB for metadata
        self.total_memory_mb += pattern_size_mb
        
        # Log memory usage periodically
        if self.next_id % 1000 == 0:
            logging.info(f"FAISS memory usage: {self.total_memory_mb:.2f} MB, Patterns: {self.next_id}")
            
        # Store in memory dict
        self.memory[self.next_id] = pattern_metadata
        pattern_id = self.next_id
        self.next_id += 1
        
        # Periodically save patterns
        current_time = time.time()
        if current_time - self.last_prune_time > 300:  # Every 5 minutes
            self._save_patterns()
            self.last_prune_time = current_time
            
        return pattern_id
        
    def query_pattern(self, embedding: np.ndarray, top_k: int = 3) -> Optional[Dict[str, Any]]:
        """
        Find similar patterns with enhanced error handling and performance
        """
        try:
            if self.index.ntotal == 0:
                logging.warning("No patterns in memory to search. Suggest running warm-up.")
                return None
                
            # Ensure embedding is the correct data type
            embedding = embedding.astype(np.float32)
            
            # Check embedding dimension
            if embedding.shape != (self.dimension,):
                logging.error(f"Query embedding dimension mismatch: got {embedding.shape}, expected ({self.dimension},)")
                return None

            # Use FAISS to find similar patterns
            D, I = self.index.search(np.expand_dims(embedding, axis=0), min(top_k, self.index.ntotal))
            matches = []
            
            for dist, idx in zip(D[0], I[0]):
                if idx in self.memory:
                    result = self.memory[idx].copy()
                    
                    # Remove the stored embedding from result to save bandwidth
                    if "embedding" in result:
                        del result["embedding"]
                        
                    result['distance'] = float(dist)
                    result['pattern_id'] = int(idx)
                    matches.append(result)

            return {"matches": matches} if matches else None

        except Exception as e:
            logging.exception(f"FAISS search error: {e}")
            return None
            
    def _prune_old_patterns(self):
        """
        Prune oldest patterns to make room for new ones
        Enhanced with proper FAISS index rebuilding
        """
        logging.warning(f"FAISS reached capacity ({self.max_patterns} patterns), pruning oldest 10%")
        
        # Keep the newest 90% of patterns
        keep_count = int(self.max_patterns * 0.9)
        remove_count = self.max_patterns - keep_count
        
        # Sort patterns by ID (proxy for age)
        sorted_ids = sorted(self.memory.keys())
        
        # IDs to remove (oldest 10%)
        remove_ids = sorted_ids[:remove_count]
        
        # We'll rebuild the FAISS index with only the patterns we want to keep
        new_index = faiss.IndexFlatL2(self.dimension)
        new_memory = {}
        new_id_mapping = {}  # Maps old IDs to new IDs
        
        # Use batching for better performance
        batch_size = 1000
        embeddings = []
        ids_to_keep = []
        
        # Process in batches
        for old_id in sorted_ids[remove_count:]:
            metadata = self.memory[old_id]
            
            # Get the embedding from metadata
            if "embedding" in metadata:
                # Convert back to float32 for FAISS
                embedding = metadata["embedding"].astype(np.float32)
                embeddings.append(embedding)
                ids_to_keep.append(old_id)
                
                # Process in batches
                if len(embeddings) >= batch_size:
                    # Add embeddings to new index
                    embeddings_array = np.array(embeddings)
                    new_index.add(embeddings_array)
                    
                    # Update ID mapping
                    start_idx = new_index.ntotal - len(embeddings)
                    for i, old_id in enumerate(ids_to_keep):
                        new_id = start_idx + i
                        new_id_mapping[old_id] = new_id
                        new_memory[new_id] = self.memory[old_id]
                    
                    # Reset batch
                    embeddings = []
                    ids_to_keep = []
                    
        # Process remaining embeddings
        if embeddings:
            embeddings_array = np.array(embeddings)
            new_index.add(embeddings_array)
            
            # Update ID mapping
            start_idx = new_index.ntotal - len(embeddings)
            for i, old_id in enumerate(ids_to_keep):
                new_id = start_idx + i
                new_id_mapping[old_id] = new_id
                new_memory[new_id] = self.memory[old_id]
        
        # Replace the old index and memory
        self.index = new_index
        self.memory = new_memory
        self.next_id = new_index.ntotal
        
        # Recalculate memory usage
        self.total_memory_mb = self.total_memory_mb * (keep_count / self.max_patterns)
        
        # Try to rebuild GPU index if we were using GPU
        if self.using_gpu:
            try:
                gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                self.index = gpu_index
            except Exception as e:
                logging.error(f"Failed to rebuild GPU index after pruning: {e}")
                self.using_gpu = False
        
        logging.info(f"FAISS pruned {remove_count} patterns, new count: {self.next_id}")
        
        # Save patterns after pruning
        self._save_patterns()
            
    def _save_patterns(self):
        """Save pattern metadata to disk for persistence"""
        try:
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            # Save basic metadata (not the full embeddings)
            metadata_list = []
            for pattern_id, metadata in self.memory.items():
                meta_copy = metadata.copy()
                # Don't store the embedding in the JSON file - too large
                if "embedding" in meta_copy:
                    del meta_copy["embedding"]
                meta_copy["pattern_id"] = pattern_id
                metadata_list.append(meta_copy)
                
            with open("logs/faiss_patterns_metadata.json", "w") as f:
                json.dump(metadata_list, f)
                
            # Save embeddings in a binary file
            embeddings = []
            pattern_ids = []
            
            for pattern_id, metadata in self.memory.items():
                if "embedding" in metadata:
                    embeddings.append(metadata["embedding"].astype(np.float32))
                    pattern_ids.append(pattern_id)
                    
            if embeddings:
                embeddings_array = np.array(embeddings)
                np.save("logs/faiss_embeddings.npy", embeddings_array)
                np.save("logs/faiss_pattern_ids.npy", np.array(pattern_ids))
                
            logging.info(f"Saved {len(metadata_list)} patterns to disk")
        except Exception as e:
            logging.error(f"Failed to save patterns: {e}")
            
    def _load_patterns(self):
        """Load patterns from disk if available"""
        try:
            # Check if files exist
            if not (os.path.exists("logs/faiss_patterns_metadata.json") and 
                    os.path.exists("logs/faiss_embeddings.npy") and 
                    os.path.exists("logs/faiss_pattern_ids.npy")):
                logging.info("No saved patterns found")
                return
                
            # Load metadata
            with open("logs/faiss_patterns_metadata.json", "r") as f:
                metadata_list = json.load(f)
                
            # Load embeddings and pattern IDs
            embeddings = np.load("logs/faiss_embeddings.npy")
            pattern_ids = np.load("logs/faiss_pattern_ids.npy")
            
            # Rebuild index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.memory = {}
            
            # Add embeddings to index
            self.index.add(embeddings)
            
            # Restore memory dict
            for i, metadata in enumerate(metadata_list):
                pattern_id = metadata.get("pattern_id", i)
                
                # Find corresponding embedding
                embedding_idx = np.where(pattern_ids == pattern_id)[0]
                if len(embedding_idx) > 0:
                    metadata["embedding"] = embeddings[embedding_idx[0]]
                    
                self.memory[pattern_id] = metadata
                
                # Update next_id to be higher than any loaded pattern_id
                self.next_id = max(self.next_id, pattern_id + 1)
                
            # Try to rebuild GPU index if available
            if faiss.get_num_gpus() > 0:
                try:
                    self.gpu_resources = faiss.StandardGpuResources()
                    gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                    self.index = gpu_index
                    self.using_gpu = True
                except Exception as e:
                    logging.warning(f"Failed to build GPU index on load: {e}")
                    self.using_gpu = False
                    
            # Estimate memory usage
            self.total_memory_mb = (embeddings.nbytes + len(json.dumps(metadata_list)) + 1024 * 1024) / (1024 * 1024)
            
            logging.info(f"Loaded {len(self.memory)} patterns from disk")
        except Exception as e:
            logging.error(f"Failed to load patterns: {e}")

# Singleton Memory Bank Instance
pattern_memory_bank = PatternMemoryBank(dimension=4096)

# Function to rebuild the index with custom parameters
def rebuild_index(dimension: int = 4096, use_gpu: bool = True):
    """
    Rebuild the FAISS index with custom parameters.
    Useful for performance tuning or recovery.
    """
    global pattern_memory_bank
    
    try:
        # Save existing patterns
        pattern_memory_bank._save_patterns()
        
        # Create new memory bank
        pattern_memory_bank = PatternMemoryBank(dimension=dimension)
        
        logging.info(f"FAISS index rebuilt with dimension={dimension}, GPU={use_gpu}")
        return True
    except Exception as e:
        logging.error(f"Failed to rebuild FAISS index: {e}")
        return False

# Force warm-up during import
if pattern_memory_bank.index.ntotal == 0:
    for _ in range(10):
        dummy_embedding = np.random.rand(4096).astype(np.float32)
        pattern_memory_bank.add_pattern(dummy_embedding, {
            "pattern": "warmup",
            "confidence": 0.5
        })
    logging.info("[WARMUP] FAISS memory pre-loaded with dummy patterns.")

# Optional manual test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_embedding = np.random.rand(4096).astype(np.float32)
    result = pattern_memory_bank.query_pattern(test_embedding)
    logging.info(f"Query Result: {result}")