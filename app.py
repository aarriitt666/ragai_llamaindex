import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, StorageContext
from llama_index.core.settings import Settings
from llama_index.core.response.pprint_utils import pprint_response
import warnings
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv('.env')

# Suppress specific FutureWarnings from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub')

class QueryBundle:
    def __init__(self, query_str):
        self.query_str = query_str

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

# Initialize the reranker
reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=5)

# Initialize the parser
parser = SentenceSplitter()

def pprint_response(response, show_source=False):
    if isinstance(response, str):
        print(response)  # Handle the string directly
    else:
        if response.response is None:
            print("No response.")
        else:
            print(response.response)
            if show_source:
                print("Source:", response.source)
                
class EnhancedTextNode:
    def __init__(self, text_node):
        self.node = text_node  # Wrap the original TextNode

    def get_content(self, metadata_mode):
        return self.node.text  # Implement a method that the reranker might call

# Modify the enhance_and_rerank_responses function to wrap TextNodes
def enhance_and_rerank_responses(responses, query):
    """ Combine reranking and enhancing to select the most comprehensive and relevant response. """
    if not responses:
        return "No responses available."
    
    # Reranking using the semantic reranker
    query_bundle = QueryBundle(query)
    nodes = [EnhancedTextNode(TextNode(text=res)) for res in responses]  # Wrap TextNodes for compatibility
    reranked_nodes = reranker.postprocess_nodes(nodes=nodes, query_bundle=query_bundle)
    reranked_responses = [node.node.text for node in reranked_nodes]  # Adjust access to text

    # Enhance the response quality by selecting the most comprehensive answer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(reranked_responses)
    cosine_matrix = cosine_similarity(tfidf_matrix)
    avg_similarity = cosine_matrix.mean(axis=0)
    best_response_idx = avg_similarity.argmax()
    return reranked_responses[best_response_idx]

# Function to initialize or load the index
@st.cache_resource(show_spinner=False)
def initialize():
    if not os.path.isfile(os.path.join(storage_path, 'docstore.json')):
        documents = SimpleDirectoryReader(documents_path).load_data()
        nodes = parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=storage_path)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context)
    return index

def main():
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
                response_texts = response.response if isinstance(response.response, list) else [response.response]
                st.write(response_texts)
                
                # Create a QueryBundle object
                # query_bundle = QueryBundle(prompt)
                
                # Assume response.response is a list of strings
                # nodes = [TextNode(text=res) for res in response.response] if isinstance(response.response, list) else []

                # reranked_nodes = reranker.postprocess_nodes(nodes=nodes, query_bundle=query_bundle)
                # reranked_response = ' '.join([node.text for node in reranked_nodes])
                # st.write(reranked_response)
                best_response = enhance_and_rerank_responses(response_texts, prompt)
                pprint_response(best_response, show_source=True)
                st.session_state.messages.append({'role': 'assistant', 'content': best_response})
    else:
        st.write("Upload documents to start using the application.") 
        
if __name__ == "__main__":
    main()           
    