import sys
import os
import pandas as pd

# Fix for Windows Console Encoding
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluation.research_benchmark import ResearchBenchmark
from src.core.indexing import SearchIndex
from src.core.generator import get_generator
from src.core.ingestion import DocumentProcessor

def run_experiments():
    print("Initializing Native Experimental Rigor Pipeline (Zero-LangChain)...")
    
    # 1. Setup
    data_dir = os.path.join(project_root, "data")
    
    # Initialize Native Generator (NVIDIA NIM)
    llm = get_generator()
    
    # 2. Process Documents Natively
    proc = DocumentProcessor()
    chunks = proc.process_directory(data_dir, use_semantic=True)
    
    # 3. Build Native Index (FAISS + BM25)
    index = SearchIndex()
    index.build_indexes(chunks)
    
    # 4. Define Evaluation Set
    import json
    # Updated path to new location
    questions_path = os.path.join(project_root, "data", "evaluation", "TEST_QUESTIONS.json")
    if os.path.exists(questions_path):
        with open(questions_path, "r", encoding="utf-8") as f:
            test_queries = json.load(f)
            # Adapt structure if needed (ensure 'q' key exists)
            print(f"Loaded {len(test_queries)} questions from {questions_path}")
    else:
        print(f"Warning: {questions_path} not found. Using defaults.")
        test_queries = [
            {"q": "Ի՞նչ է ասվում դատական ակտերի մասին:", "source": "ԴԱՏԱԿԱՆ", "expected": ""},
        ]

    # 5. Run Benchmark
    bench = ResearchBenchmark(index, llm)
    summary_report, detailed_report = bench.run_strategy_comparison(test_queries)
    
    print("\nBENCHMARK RESULTS (Summary)")
    print("=" * 40)
    # Check if to_markdown is available (needs tabulate)
    try:
        print(summary_report.to_markdown())
    except:
        print(summary_report)
    print("=" * 40)
    
    # 6. Save for report
    results_dir = os.path.join(project_root, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    summary_path = os.path.join(results_dir, "research_results_summary.csv")
    detailed_path = os.path.join(results_dir, "research_results_detailed.csv")
    
    summary_report.to_csv(summary_path, index=False)
    detailed_report.to_csv(detailed_path, index=False)
    print(f"Summary saved to: {summary_path}")
    print(f"Detailed per-question results saved to: {detailed_path}")

if __name__ == "__main__":
    run_experiments()
