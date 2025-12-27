import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
import openai

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Job Shop Lean Chatbot", page_icon="🏭", layout="centered")
st.title("🏭 Job Shop Lean: Multimodal RAG")

# --- Sidebar & Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    # OpenAI API Key
    openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY") or "")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        openai.api_key = openai_key
    
    # LlamaCloud API Key
    llama_key = st.text_input("LlamaCloud API Key", type="password", value=os.getenv("LLAMA_CLOUD_API_KEY") or "")
    if llama_key:
        os.environ["LLAMA_CLOUD_API_KEY"] = llama_key

    if not openai_key or not llama_key:
        st.warning("Please provide both API Keys to proceed.")
        st.info("Get LlamaCloud Key: https://cloud.llamaindex.ai")
        st.stop()
    else:
        st.success("Keys loaded!")

# --- Global Settings ---
# Text LLM for standard queries
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
# Multimodal LLM for visual queries
multi_modal_llm = OpenAIMultiModal(model="gpt-4o", max_new_tokens=1000)

# --- Data Loading & Indexing ---
@st.cache_resource(show_spinner=False)
def load_data():
    PERSIST_DIR = "./storage_multimodal"
    
    if not os.path.exists(PERSIST_DIR):
        with st.spinner("Parsing PDF with LlamaParse... This may take a while."):
            # Configure parser for markdown (tables/charts)
            parser = LlamaParse(
                result_type="markdown",
                verbose=True
            )
            
            file_extractor = {".pdf": parser}
            
            # Load documents
            reader = SimpleDirectoryReader(
                input_files=["MY_BOOK_INCLUDING_APPENDICES.pdf"],
                file_extractor=file_extractor
            )
            documents = reader.load_data()
            st.toast(f"Parsed {len(documents)} context chunks.", icon="📄")
            
            # Create MultiModal Index
            index = MultiModalVectorStoreIndex.from_documents(
                documents,
                multi_modal_llm=multi_modal_llm
            )
            
            # Persist
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            st.toast("Index created and saved!", icon="💾")
            return index
    else:
        with st.spinner("Loading index from storage..."):
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(
                storage_context,
                image_store=None # Simplify for now, focus on text/markdown
            )
            return index

try:
    index = load_data()
except Exception as e:
    st.error(f"Error loading index: {e}")
    st.stop()

# --- Chat Engine Setup ---
if "chat_engine" not in st.session_state:
    # We default to the context chat engine, but we'll manually route visual queries
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context",
        system_prompt="You are an expert on 'Job Shop Lean'. Use context to answer. If asked about figures or tables, describe them in detail."
    )

# --- Query Routing Logic ---
def is_visual_query(query):
    keywords = ["figure", "fig", "table", "chart", "diagram", "layout", "map", "gantt", "spreadsheet", "image"]
    return any(word in query.lower() for word in keywords)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me about Job Shop Lean! I can explain figures, tables, and text."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Hybrid Routing
            if is_visual_query(prompt):
                # Use Multimodal capability if keywords detected
                # Note: Currently assumes index has image nodes or markdown description
                # GPT-4o via LlamaParse markdown acts as "multimodal" interpretation of the text representation of images
                 response = st.session_state.chat_engine.chat(prompt)
                 # Ideally we would use retrieval_mode="image" but LlamaParse converts to markdown text mainly.
                 # If we want pure image retrieval we need specific image nodes. 
                 # For now, relying on LlamaParse's superior markdown description for GPT-4o to reason over.
            else:
                response = st.session_state.chat_engine.chat(prompt)
            
            st.markdown(response.response)
            
    st.session_state.messages.append({"role": "assistant", "content": response.response})
