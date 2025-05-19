import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

if __name__ == "__main__":
    # Example usage (for testing)
    # This part would typically be handled by the Streamlit app
    # Need to have a chroma_db directory with embeddings for this to work
    # For a full test, you would first run document_processor.py to create the db

    # Create a dummy chroma_db directory and a dummy vectorstore for testing purposes
    if not os.path.exists("./chroma_db"):
        os.makedirs("./chroma_db")
        # In a real scenario, you would embed documents here
        # For this test, we'll just create an empty Chroma instance
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


    vectorstore = get_vectorstore()
    if vectorstore:
        chatbot = get_chatbot(vectorstore)
        print("Chatbot initialized. Type 'quit' to exit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
            response = chatbot({"question": user_input})
            print(f"Bot: {response['answer']}")
    else:
        print("Vectorstore not found. Please process documents first.")

    # Clean up dummy chroma_db directory
    # import shutil
    # if os.path.exists("./chroma_db"):
    #     shutil.rmtree("./chroma_db")
