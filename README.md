# RAG System with Mistral-7B

This project implements a Retrieval-Augmented Generation (RAG) system using the Mistral-7B model through Ollama. It's designed to process and answer questions about scientific papers, with a focus on maintaining conversation context and providing streaming responses.

## Features

- PDF text extraction and processing
- Chunk-based text splitting for efficient retrieval
- Vector storage using Chroma DB
- Conversation memory for context-aware responses
- Word-by-word streaming responses
- Interactive question-answering mode

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- Mistral-7B model pulled in Ollama

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

## Usage

1. Place your PDF file in the project directory (default name: `s41587-024-02551-2.pdf`)

2. Run the script:

```bash
python main.py
```

3. The system will:

   - Process the PDF file
   - Create text chunks
   - Initialize the vector store
   - Run through example questions
   - Enter interactive mode

4. In interactive mode:
   - Type your questions about the paper
   - Get streaming responses
   - Type 'quit' to exit

## Project Structure

- `main.py`: Main script containing the RAG implementation
- `requirements.txt`: Project dependencies
- `chroma_db/`: Directory for vector store persistence
- `*.pdf`: Input PDF files

## Dependencies

The project uses several key libraries:

- langchain: For RAG implementation
- langchain-huggingface: For embeddings
- langchain-ollama: For LLM integration
- PyPDF2: For PDF processing
- Chroma: For vector storage

## Notes

- The system uses the Mistral-7B model through Ollama for text generation
- Responses are streamed word by word for better readability
- Conversation history is maintained during the session
- The vector store is persisted in the `chroma_db` directory
