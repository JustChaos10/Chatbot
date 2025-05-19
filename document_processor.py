import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def load_document(file_path):
    """Loads a document based on its file extension."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    return loader.load()

def split_documents(documents):
    """Splits documents into smaller chunks."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def embed_documents(documents, persist_directory="./chroma_db"):
    """Embeds document chunks and stores them in a vector database."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    return vectorstore

if __name__ == "__main__":
    # Example usage (for testing)
    # Create a dummy text file for testing
    with open("test_document.txt", "w") as f:
        f.write("This is a test document. It contains some information.")

    documents = load_document("test_document.txt")
    split_docs = split_documents(documents)
    vectorstore = embed_documents(split_docs)
    print(f"Created vectorstore with {len(split_docs)} chunks.")

    # Clean up dummy file
    os.remove("test_document.txt")
