# Local Document Analyzer

A powerful, locally-hosted document analysis system that provides document processing, querying, and summarization capabilities using state-of-the-art AI models. The system is designed to run entirely on your local machine, ensuring data privacy and security.

## Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)

## Features

### Core Capabilities
- Multi-format document processing (PDF, DOCX, XLSX, CSV, HTML, MD, TXT)
- Efficient document embedding and retrieval using FAISS
- AI-powered question answering
- Automatic document summarization
- Advanced text preprocessing
- Local model execution
- Web-based user interface

### Performance Features
- Embedding caching system
- GPU acceleration support
- Chunked document processing
- Efficient vector similarity search

### Security Features
- Fully local execution
- No external API dependencies
- Data privacy by design
- Secure file handling

## System Architecture

### High-Level Architecture
```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Web Interface  │────▶│ Document Analyzer │────▶│ Model Pipeline │
└─────────────────┘     └──────────────────┘     └────────────────┘
         │                       │                        │
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│   File Upload   │     │  Document Cache   │     │  FAISS Index   │
└─────────────────┘     └──────────────────┘     └────────────────┘
```

### Component Breakdown

1. **Document Processor**
   - File format handling
   - Text extraction
   - Preprocessing pipeline
   - Document chunking

2. **Embedding System**
   - SentenceTransformer model
   - Caching mechanism
   - FAISS indexing
   - Similarity search

3. **AI Models**
   - Question answering (RoBERTa)
   - Summarization (BART)
   - Embedding generation
   - Context processing

4. **Web Interface**
   - File upload
   - Query interface
   - Results display
   - Document management

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional)
- 8GB+ RAM
- 2GB+ disk space

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/local-doc-analyzer.git
cd local-doc-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements
```txt
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4  # or faiss-gpu
flask>=2.0.0
pypdf>=3.0.0
python-docx>=0.8.11
pandas>=1.5.0
openpyxl>=3.0.0
beautifulsoup4>=4.9.0
markdown>=3.4.0
tqdm>=4.65.0
numpy>=1.23.0
```

## Usage

### Starting the Server
```bash
python main.py
```

### Web Interface
Access the web interface at `http://localhost:5000`

1. **Document Upload**
   - Click "Upload Documents"
   - Select one or more supported files
   - Wait for processing completion

2. **Querying Documents**
   - Enter your question in the text area
   - Click "Ask" to get answers
   - View source documents and confidence scores

3. **Viewing Summaries**
   - Access the "Document Summaries" section
   - View auto-generated summaries
   - Navigate through processed documents

### Python API

```python
from doc_analyzer import LocalDocumentAnalyzer

# Initialize analyzer
analyzer = LocalDocumentAnalyzer(
    embedding_model="all-MiniLM-L6-v2",
    cache_dir=".cache"
)

# Process documents
analyzer.process_documents([
    "path/to/doc1.pdf",
    "path/to/doc2.docx"
])

# Ask questions
result = analyzer.answer_question(
    "What is the main conclusion?"
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
```

## API Reference

### LocalDocumentAnalyzer Class

#### Methods

`__init__(embedding_model: str = "all-MiniLM-L6-v2", cache_dir: str = ".cache", device: str = None)`
- Initializes the document analyzer
- Parameters:
  - embedding_model: Model name for embeddings
  - cache_dir: Directory for caching
  - device: 'cuda' or 'cpu'

`load_document(file_path: str) -> Document`
- Loads and processes a document
- Returns: Document object

`process_documents(file_paths: List[str], chunk_size: int = 512)`
- Processes multiple documents
- Parameters:
  - file_paths: List of file paths
  - chunk_size: Size of text chunks

`answer_question(query: str, k: int = 3) -> Dict`
- Answers questions about documents
- Returns: Dictionary with answer, confidence, and sources

## Configuration

### Environment Variables
```env
CACHE_DIR=.cache
MODEL_DEVICE=cuda
EMBEDDING_MODEL=all-MiniLM-L6-v2
PORT=5000
HOST=0.0.0.0
```

### Model Configuration
```python
config = {
    'chunk_size': 512,
    'max_length': 150,
    'top_k': 3,
    'cache_enabled': True
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit pull request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Maintain type hints

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   - Reduce batch size
   - Use smaller models
   - Enable gradient checkpointing

2. **Slow Processing**
   - Check cache configuration
   - Optimize chunk size
   - Use GPU acceleration

3. **File Format Errors**
   - Verify file compatibility
   - Check file encoding
   - Update format handlers

### Support

For issues and feature requests, please use the GitHub issue tracker or contact the maintainers.

