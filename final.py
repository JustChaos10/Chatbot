import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# document_processor.py functions
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

# chatbot.py functions and templates
# Custom prompt template for combining documents
combine_docs_prompt = PromptTemplate.from_template(
    """You are a helpful AI assistant. You have been provided with content from TWO distinct documents, labeled with their source files (e.g., Source: B_Resume.pdf and Source: S_Resume.pdf).
Your task is to answer the user's question based on the provided context, drawing information from BOTH documents.
The user has indicated that there is a difference between these two documents.
If the user asks to compare or contrast the documents, carefully analyze the content from EACH source file and identify the key differences and similarities, paying close attention to any distinctions.
Ensure you consider information from both documents when making comparisons.
Focus on comparing relevant sections like experience, skills, education, and projects if they are present in the documents.
If the context is not relevant to the question, use your general knowledge to answer.
If you cannot answer the question from either the context or your general knowledge, say that you don't know.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:"""
)

# Custom document prompt to include source
document_prompt = PromptTemplate.from_template("Source: {source}\n{page_content}")

# Prompt for generating a standalone question
question_generator_prompt = PromptTemplate.from_template(
    """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    {input}
    Standalone question:"""
)

def get_chatbot(vectorstore):
    """Initializes and returns the conversational retrieval chatbot."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", google_api_key=GEMINI_API_KEY, temperature=0.7)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # Create a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, question_generator_prompt
    )

    # Create a chain to combine documents and answer
    # This chain formats the retrieved documents and passes them as 'context' to the final prompt.
    combine_docs_chain = (
        {
            "context": history_aware_retriever | (lambda docs: "\n\n".join([document_prompt.format(page_content=doc.page_content, source=doc.metadata['source']) for doc in docs])),
            "chat_history": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
        | combine_docs_prompt
        | llm
        | StrOutputParser()
    )

    # Create the main retrieval chain
    # Note: create_retrieval_chain expects the history_aware_retriever and the combine_docs_chain.
    # The input to this chain should be {"input": user_question, "chat_history": chat_history}.
    retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

    # Integrate memory with the chain
    # Note: This is a simplified integration. For more complex memory scenarios,
    # refer to LangChain documentation on RunnableWithHistory.
    def get_session_history(session_id: str):
        # In a real Streamlit app, you would manage session history per user session.
        # For this example, we'll use a simple in-memory approach tied to the Streamlit session state.
        # This requires access to st.session_state, which is not directly available here.
        # A more robust solution would involve passing session state or a memory object from app.py.
        # For demonstration, we'll return a dummy memory.
        # To fully integrate, you would need to pass the actual memory object from app.py.
        return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


    # This part needs to be handled in app.py where st.session_state is accessible.
    # The chain itself doesn't directly manage the Streamlit session state memory.
    # We will return the retrieval_chain and handle memory in app.py.

    return retrieval_chain

def get_vectorstore(persist_directory="./chroma_db"):
    """Loads an existing vectorstore or returns None if it doesn't exist."""
    if os.path.exists(persist_directory):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectorstore
    return None

# app.py main logic
# Set up the Streamlit page
st.set_page_config(page_title="Hybrid Chatbot", layout="wide")
st.title("Hybrid Chatbot with Document Upload")

# Directory to save uploaded documents
UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Directory for Chroma DB
CHROMA_DB_DIR = "./chroma_db"

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for document upload
with st.sidebar:
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded {uploaded_file.name}")

            # Process and embed the document
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    documents = load_document(file_path)
                    split_docs = split_documents(documents)

                    # Load existing vectorstore or create a new one
                    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
                         st.session_state.vectorstore = embed_documents(split_docs, persist_directory=CHROMA_DB_DIR)
                    else:
                        # Add documents to the existing vectorstore
                        # Assuming Chroma has an add_documents method
                        st.session_state.vectorstore.add_documents(split_docs)

                    st.success(f"Document {uploaded_file.name} processed and embedded successfully!")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

# Load existing vectorstore if available
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_vectorstore(persist_directory=CHROMA_DB_DIR)

# Initialize chatbot if vectorstore is available
if st.session_state.vectorstore:
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = get_chatbot(st.session_state.vectorstore)
else:
    st.info("Please upload a document to start chatting.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot response
    if st.session_state.vectorstore and st.session_state.chatbot:
        with st.spinner("Getting response..."):
                try:
                    response = st.session_state.chatbot.invoke({"input": prompt, "chat_history": st.session_state.messages})
                    bot_response = response["answer"]
                    # Add bot response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    # Display bot response
                    with st.chat_message("assistant"):
                        st.markdown(bot_response)
                except Exception as e:
                    st.error(f"Error getting chatbot response: {e}")
    else:
        st.warning("Please upload and process a document first.")
