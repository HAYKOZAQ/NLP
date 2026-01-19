import sys
import os

# Fix for Windows Console Encoding
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.ingestion import DocumentProcessor
from src.core.indexing import SearchIndex

def run_indexing():
    print("Legal Indexing Engine: Processing New Documents...")
    data_dir = os.path.join(project_root, "data")
    index_path = os.path.join(project_root, "index")
    
    # 1. Process PDFs
    print(f"Scanning directory: {data_dir}")
    proc = DocumentProcessor()
    chunks = proc.process_directory(data_dir, use_semantic=True)
    
    if not chunks:
        print("Error: No documents found to index.")
        return

    print(f"Total chunks created: {len(chunks)}")
    
    # 2. Build Index
    index = SearchIndex()
    index.build_indexes(chunks)
    
    # 3. Save to disk for high-speed app loading
    print(f"Saving persistent index to: {index_path}")
    index.save_indexes(index_path)
    print("Indexing Complete. The Legal RAG is now up-to-date.")

if __name__ == "__main__":
    run_indexing()
