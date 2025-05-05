from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import PyPDF2
import sys
import os
import time

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from each page
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num + 1}: {str(e)}")
                    continue
            
            if not text.strip():
                raise Exception("No text could be extracted from the PDF")
            
            return text
    except Exception as e:
        print(f"Error reading PDF file: {str(e)}")
        sys.exit(1)

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

try:
    # Read the PDF file
    pdf_path = "s41587-024-02551-2.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found. Please make sure the file exists in the current directory.")
        sys.exit(1)
    
    print("Reading PDF file...")
    text = extract_text_from_pdf(pdf_path)
    print("PDF reading completed successfully.")
    
    # Split the text into chunks
    print("Splitting text into chunks...")
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} text chunks.")

    # Initialize the embedding model
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create the vector store
    print("Creating vector store...")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Vector store created successfully.")

    # Initialize the LLM (Mistral through Ollama)
    try:
        print("Initializing LLM...")
        llm = OllamaLLM(
            model="mistral",
            streaming=True  # Enable streaming
        )
        print("LLM initialized successfully.")
    except Exception as e:
        print("Error: Could not initialize Mistral model. Please make sure Ollama is running and the model is installed.")
        print("You can install the model by running: ollama pull mistral")
        sys.exit(1)

    # Create a custom prompt template
    template = """You are having a conversation about a scientific paper. Use the following pieces of context and conversation history to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Conversation History:
    {chat_history}

    Question: {question}

    Answer: Let me help you with that."""

    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "chat_history", "question"]
    )

    # Initialize conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create the conversational retrieval chain
    print("Setting up RAG system...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    print("RAG system setup completed.\n")

    def query_rag(question: str) -> str:
        """Query the RAG system with a question."""
        try:
            # Use a callback to handle streaming
            full_response = []
            for chunk in qa_chain.stream({"question": question}):
                if "answer" in chunk:
                    # Split the chunk into words and stream each word
                    words = chunk["answer"].split()
                    for word in words:
                        print(word + " ", end="", flush=True)
                        time.sleep(0.05)  # Small delay between words for readability
                        full_response.append(word + " ")
            print()  # Add a newline after the response
            return "".join(full_response)
        except Exception as e:
            return f"Error processing question: {str(e)}"

    if __name__ == "__main__":
        # Example questions about the paper on engineered adipocytes
        questions = [
            "What is the main finding of this study about engineered adipocytes and tumor progression?",
            "How do the engineered adipocytes work to suppress tumor progression?",
            "What types of cancer models were used in this study?",
            "What are the potential clinical implications of this research?",
            "What are the key limitations or challenges mentioned in the study?"
        ]
        
        print("RAG System Test\n")
        for question in questions:
            print(f"Q: {question}")
            print("A: ", end="", flush=True)  # Print "A: " without newline
            answer = query_rag(question)
            print()  # Add a newline after the response
        
        # Interactive mode
        print("\n=== Interactive Mode ===")
        print("You can now ask questions about the paper. Type 'quit' to exit.")
        print("The system will remember our conversation context.\n")
        
        while True:
            user_question = input("Your question: ").strip()
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the RAG system!")
                break
            
            if not user_question:
                print("Please enter a valid question.")
                continue
                
            print("\nProcessing your question...\n")
            print("Answer: ", end="", flush=True)  # Print "Answer: " without newline
            answer = query_rag(user_question)
            print()  # Add a newline after the response

except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
    sys.exit(1)
