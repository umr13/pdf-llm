# RAG System with Mistral-7B

This project implements a Retrieval-Augmented Generation (RAG) system using the Mistral-7B model through Ollama. It's designed to process and answer questions about scientific papers, with a focus on maintaining conversation context and providing streaming responses.

## Features

- PDF text extraction and processing
- Support for multiple PDF files
- Chunk-based text splitting for efficient retrieval
- Vector storage using Chroma DB
- Conversation memory for context-aware responses
- Word-by-word streaming responses
- Interactive question-answering mode

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- Mistral-7B model pulled in Ollama
- Sufficient disk space for PDF files and vector store
- Minimum 8GB RAM recommended

## Installation

1. Clone the repository

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Pull the Mistral model in Ollama:

```bash
ollama pull mistral
```

## Project Structure

```
.
├── pdf_files/          # Directory for PDF files
├── chroma_db/          # Vector store directory
├── main.py            # Main script
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Usage

1. Place your PDF files in the `pdf_files` directory:

```bash
cp your_paper.pdf pdf_files/
```

2. Run the script:

```bash
python main.py
```

3. The system will:

   - Process all PDF files in the `pdf_files` directory
   - Create text chunks
   - Initialize the vector store
   - Run through example questions
   - Enter interactive mode

4. In interactive mode:
   - Type your questions about the papers
   - Get streaming responses
   - Type 'quit' to exit

## Limitations

1. PDF Processing:

   - Only processes text content; images, tables, and figures are ignored
   - May not handle complex PDF layouts correctly
   - Limited support for mathematical equations and special characters
   - Maximum file size depends on available memory

2. Memory and Performance:

   - Requires significant RAM for processing large PDFs
   - Vector store size grows with the number of documents
   - Processing time increases with document size and complexity
   - Streaming response speed depends on system resources

3. Model Limitations:

   - Responses are based on Mistral-7B's training data cutoff
   - May not understand highly technical or domain-specific content
   - Limited context window size affects response quality
   - No support for real-time updates or external data sources

4. File Management:

   - No automatic file organization or categorization
   - No support for file versioning or updates
   - Limited metadata tracking
   - No built-in backup system

5. Security:
   - No authentication or access control
   - No encryption for stored data
   - No input validation for PDF files
   - No protection against malicious files

## Dependencies

The project uses several key libraries:

- langchain: For RAG implementation
- langchain-huggingface: For embeddings
- langchain-ollama: For LLM integration
- PyPDF2: For PDF processing
- Chroma: For vector storage

## Troubleshooting

1. If Ollama is not running:

```bash
ollama serve
```

2. If the Mistral model is not found:

```bash
ollama pull mistral
```

3. If PDF processing fails:

   - Check if the PDF is text-based (not scanned)
   - Ensure the PDF is not password-protected
   - Verify the PDF is not corrupted

4. If memory issues occur:
   - Reduce the number of PDF files
   - Decrease chunk size in the code
   - Close other memory-intensive applications
