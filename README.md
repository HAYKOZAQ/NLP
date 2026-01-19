# Better Orin AI: Advanced Legal RAG Assistant

**Better Orin AI** is a cutting-edge Retrieval-Augmented Generation (RAG) system designed to serve as a high-precision legal assistant for Armenian Law. Built with a focus on accuracy and trust, it features a **Hybrid Retrieval Engine**, a novel **Real-Time Hallucination Guard**, and a sleek **Gemini-style User Interface**.

Developed by **Hayk Zakaryan** and **Robert Kocharyan**.

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 🌟 Key Features

### 🧠 Core Intelligence

- **Hybrid Retrieval System**: Combines semantic search (Sentence Transformers) with keyword search (BM25) for high-recall, high-precision document recovery.
- **RAG Architecture**: Uses retrieval results to ground LLM responses (Gemini 3.0 Flash) in actual legal texts.
- **Armenian Language Support**: Optimized for processing and generating Armenian legal content.

### 🛡️ Safety & Reliability

- **Hallucination Guard**: A live verification system that analyzes the generated answer against retrieved context.
- **Real-Time Scoring**: The UI displays a "Groundedness Score" for every answer, warning users if the model is unsure or unsupported by the text.

### 💻 Modern Experience

- **Premium UI**: A responsive, dark-mode interface inspired by modern AI tools.
- **Interactive Visualizations**: Animated response generation and dynamic confidence bars.
- **Fast Indexing**: Persistent vector storage (FAISS) enables instant startup after the first run.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- A Google Cloud API Key (for Gemini models)

### 1. Installation

Clone the repository and install dependencies:

```bash
# Install required Python packages
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory (or use the existing one) and add your Google API key:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

### 3. Running the Application

Launch the web interface:

```bash
python app.py
```

The application will start at `http://localhost:5151`.
*Note: On the first run, the system will process all PDF documents in the `data/` directory. This may take a few minutes.*

---

## 📂 Project Structure

```text
better-orin-ai/
├── app.py                 # Main Flask Application & UI
├── requirements.txt       # Project Dependencies
├── .env                   # Environment Variables (API Keys)
├── data/                  # 📂 PDF Documents to be indexed
├── index/                 # 📂 Persisted FAISS Index & Metadata
├── scripts/               # 🛠️ Utility Scripts
├── src/
│   ├── core/              # Core Logic (Ingestion, Retrieval, Generation)
│   └── evaluation/        # Benchmarking Tools
├── latex/                 # 📄 Research Paper Source (LaTeX)
└── tests/                 # Unit Tests
```

---

## 📊 Evaluation & Benchmarking

To measure the system's performance (faithfulness, answer relevance, context recall), run the benchmark suite:

```bash
python src/evaluation/run_benchmark.py
```

Results are saved to the `evaluation_results/` directory.

---

## 📝 Building the Research Paper

The project includes a comprehensive academic report in LaTeX. To generate the PDF:

**Requirements**: A LaTeX distribution like MiKTeX or TeX Live.

```bash
cd latex
xelatex main
bibtex main
xelatex main
xelatex main
```

The compiled `main.pdf` will contain the full project documentation and methodology.

---

## 🏗️ Managing Data

To add new laws or legal codes:

1. Place standard PDF files into the `data/` folder.
2. Run the update script:

   ```bash
   python scripts/update_index.py
   ```

The script will scan the folder, process new documents, and rebuild the index without needing to restart the entire application server.

---

*Better Orin AI — Bringing Clarity to Law with Artificial Intelligence.*
