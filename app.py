import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, StorageContext
from llama_index.core.settings import Settings
from llama_index.core.response.pprint_utils import pprint_response

# Load environment variables
load_dotenv('.env')

# Define paths
storage_path = './vectorstore'
documents_path = './documents'

# Set the model configuration
Settings.llm = OpenAI(model='gpt-3.5-turbo', temperature=0.1)

# Ensure directories exist
if not os.path.exists(storage_path):
    os.makedirs(storage_path, exist_ok=True)

if not os.path.exists(documents_path):
    os.makedirs(documents_path, exist_ok=True)

# Function to initialize or load the index
@st.cache_resource(show_spinner=False)
def initialize():
    if not os.path.isfile(os.path.join(storage_path, 'docstore.json')):
        documents = SimpleDirectoryReader(documents_path).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=storage_path)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context)
    return index

# Check for documents and possibly upload new ones
if not os.listdir(documents_path):
    st.error("No documents found. Please upload your documents.")
    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=['pdf', 'txt', 'docx'])
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(documents_path, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getvalue())
        st.rerun()  # Rerun the script after files are uploaded

# Initialize index if documents are present
if os.listdir(documents_path):
    index = initialize()

    st.title('Ask the Document')
    if 'messages' not in st.session_state:
        st.session_state.messages = [{'role': 'assistant', 'content': 'Ask me a question!'}]

    chat_engine = index.as_chat_engine(chat_mode='condense_question', verbose=True)

    if prompt := st.text_input('Your question'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})

    for message in st.session_state.messages:
        with st.expander(f"{message['role'].title()} says:"):
            st.write(message['content'])

    if st.session_state.messages[-1]['role'] != 'assistant':
        with st.spinner('Thinking...'):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            pprint_response(response, show_source=True)
            st.session_state.messages.append({'role': 'assistant', 'content': response.response})
else:
    st.write("Upload documents to start using the application.")
            