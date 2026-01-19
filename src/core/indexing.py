import os
import pickle
import numpy as np
from typing import List, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss


class SearchIndex:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.bm25 = None
        self.faiss_index = None
        self.chunks = []

    def build_indexes(self, chunks: List[Dict]):
        """Builds both BM25 and FAISS indexes."""
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        # Build BM25
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Build FAISS
        print("Encoding chunks for FAISS...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(np.array(embeddings).astype("float32"))

        print("Indexes built successfully.")

    def save_indexes(self, path: str):
        """Saves indexes to disk."""
        if not os.path.exists(path):
            os.makedirs(path)

        # Save chunks
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

        # Save BM25
        with open(os.path.join(path, "bm25.pkl"), "wb") as f:
            pickle.dump(self.bm25, f)

        # Save FAISS
        faiss.write_index(self.faiss_index, os.path.join(path, "faiss.index"))

    def load_indexes(self, path: str):
        """Loads indexes from disk."""
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)

        with open(os.path.join(path, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)

        self.faiss_index = faiss.read_index(os.path.join(path, "faiss.index"))

    def search_bm25(self, query: str, k: int = 5) -> List[Dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top_n]

    def search_semantic(self, query: str, k: int = 5) -> List[Dict]:
        query_embedding = self.model.encode([query])
        distances, indices = self.faiss_index.search(
            np.array(query_embedding).astype("float32"), k
        )
        return [self.chunks[i] for i in indices[0] if i != -1]


if __name__ == "__main__":
    # This would be used after ingestion
    pass
