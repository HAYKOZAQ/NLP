import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.core.retrieval import HybridRetriever
from src.core.indexing import SearchIndex
from src.evaluation.research_benchmark import HallucinationChecker

class TestRAGComponents(unittest.TestCase):
    
    def setUp(self):
        # Mock index for testing
        self.index = SearchIndex()
        # Create dummy chunks
        dummy_chunks = [
            {"text": "Հոդված 1. Հայաստանի Հանրապետությունը ինքնիշխան պետություն է:", "metadata": {"source": "Constitution"}},
            {"text": "Հոդված 3. Մարդը բարձրագույն արժեք է:", "metadata": {"source": "Constitution"}}
        ]
        self.index.build_indexes(dummy_chunks)
        self.retriever = HybridRetriever(self.index, use_reranker=False) # Disable reranker for speed in unit tests

    def test_retrieval_basic(self):
        """Test if retriever returns results for exact keyword match."""
        results = self.retriever.retrieve("ինքնիշխան", k=1)
        self.assertTrue(len(results) > 0)
        self.assertIn("ինքնիշխան", results[0]["text"])

    def test_retrieval_empty(self):
        """Test graceful handling of garbage queries."""
        results = self.retriever.retrieve("sdlkfjsdlkfjsdklfjdslkfj", k=5)
        # Should return empty or low score, but not crash
        self.assertIsInstance(results, list)

    def test_evaluator_groundedness(self):
        """Test the automated groundedness checker."""
        checker = HallucinationChecker(None)
        if checker.nli_model:
            # Positive case
            res_pos = checker.get_hallucination_score("It is raining.", "It is raining outside today.")
            self.assertGreater(res_pos["groundedness_score"], 0.8)
            
            # Negative case
            res_neg = checker.get_hallucination_score("It is sunny.", "It is raining outside today.")
            self.assertLess(res_pos["groundedness_score"], 0.5)
        else:
            print("Skipping NLI test (model not loaded)")

if __name__ == '__main__':
    unittest.main()
