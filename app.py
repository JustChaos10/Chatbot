import streamlit as st
import os
from document_processor import load_document, split_documents, embed_documents
from chatbot import get_chatbot, get_vectorstore

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
