import sys
import os

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List, Dict
from src.core.indexing import SearchIndex


from sentence_transformers import CrossEncoder


class HybridRetriever:
    def __init__(self, index: SearchIndex, use_reranker: bool = True, llm=None):
        self.index = index
        self.use_reranker = use_reranker
        self.llm = llm
        self.reranker = None

        if self.use_reranker:
            try:
                # More robust cross-platform model
                print("🧠 Loading Multilingual Cross-Encoder Re-ranker...")
                self.reranker = CrossEncoder(
                    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=512
                )
            except Exception as e:
                print(f"⚠ Could not load Re-ranker: {e}")
                self.use_reranker = False

    def rewrite_query(self, query: str) -> List[str]:
        """Legal Query Expansion: Generate variations to improve recall."""
        if not self.llm or len(query.split()) > 15:
            return [query]

        prompt = f"Rewrite this legal question in Armenian in 2 different ways to help search a database. Return ONLY the questions separated by newlines. Question: {query}"
        try:
            res = self.llm.invoke(prompt)
            variants = [q.strip() for q in res.split("\n") if q.strip()]
            return [query] + variants[:1]  # Use original + 1 variant for speed
        except:
            return [query]

    def retrieve(
        self, query: str, k: int = 20, semantic_weight: float = 0.7
    ) -> List[Dict]:
        """
        Performs hybrid search + Optional Cross-Encoder Re-ranking.
        Strategy: Fetch 3x candidates (Speed) -> Re-rank top k (Precision).
        """
        if not self.index or not self.index.bm25 or not self.index.faiss_index:
            return []

        # 0. Smart Expansion
        initial_k = k * 2
        query_variants = self.rewrite_query(query)
        all_bm25 = []
        all_semantic = []

        for q in query_variants:
            all_bm25.extend(self.index.search_bm25(q, k=initial_k))
            all_semantic.extend(self.index.search_semantic(q, k=initial_k))

        # 2. Merge Scores (RRF or simple Weighted Sum)
        combined = {}

        # Semantic weights
        for i, res in enumerate(all_semantic):
            text = res["text"]
            if text not in combined:
                combined[text] = res.copy()
                # Normalize rank score (1.0 at top, 0.0 at bottom)
                combined[text]["hybrid_score"] = (
                    1.0 - i / (len(all_semantic) or 1)
                ) * semantic_weight
                
        for i, res in enumerate(all_bm25):
            text = res["text"]
            if text in combined:
                combined[text]["hybrid_score"] += (1.0 - i / (len(all_bm25) or 1)) * (
                    1.0 - semantic_weight
                )
            else:
                combined[text] = res.copy()
                combined[text]["hybrid_score"] = (1.0 - i / (len(all_bm25) or 1)) * (
                    1.0 - semantic_weight
                )

        # Convert to list
        candidates = list(combined.values())

        # 3. Re-ranking Phase
        if self.use_reranker and candidates:
            pairs = [[query, c["text"]] for c in candidates]
            scores = self.reranker.predict(pairs)

            # Attach detailed cross-encoder scores
            for i, c in enumerate(candidates):
                c["rerank_score"] = float(scores[i])

            # Sort by Re-ranker score
            sorted_results = sorted(
                candidates, key=lambda x: x["rerank_score"], reverse=True
            )
        else:
            # Fallback to Hybrid Score
            sorted_results = sorted(
                candidates, key=lambda x: x["hybrid_score"], reverse=True
            )

        return sorted_results[:k]

    def invoke(self, query: str) -> List[Dict]:
        """LangChain compatibility alias."""
        return self.retrieve(query)
