import os
import subprocess
import streamlit as st
from PIL import Image
import shutil

# Import logic
from utils import get_chroma_client, get_collection, get_embedding_model
from query.query import RAGQuerySystem
from ingestion.ingest import ingest_images, SUPPORTED_EXTS

# Constants
DATA_DIR = "./data/train"
OUTPUT_DIR = "./outputs"
SEARCH_TIMEOUT_SEC = 10  # Timeout for search button disable

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Page Config
st.set_page_config(page_title="RAG over Images", layout="wide")
st.title("RAGI: RAG over Images")

if "search_running" not in st.session_state:
    st.session_state.search_running = False

# Sidebar for Resource Loading
st.sidebar.header("System Status")
if "resources_loaded" not in st.session_state:
    st.session_state["resources_loaded"] = False


@st.cache_resource
def load_rag_system():
    st.sidebar.text("Initializing RAG System...")
    system = RAGQuerySystem()
    return system


# Load resources
with st.spinner("Loading Models..."):
    rag_system = load_rag_system()
    st.session_state["resources_loaded"] = True
st.sidebar.success("System Ready!")

# Verbose Toggle
verbose_mode = st.sidebar.toggle(
    "Verbose Mode", 
    value=False,
    disabled=st.session_state.search_running
)


# --- Ingestion Section ---
st.header("1. Ingest Images")
uploaded_files = st.file_uploader(
    "Upload images to ingest", accept_multiple_files=True, type=["jpg", "jpeg", "png"]
)

if st.button("Ingest Images"):
    if uploaded_files:
        # Save uploaded files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        st.info(f"Saved {len(uploaded_files)} images to {DATA_DIR}")
        
        # Run Ingestion
        with st.spinner("Ingesting images..."):
            # We can still use the standalone ingest function or refactor it too.
            # For now, keeping original ingest logic but maybe we should expose it via system?
            # The original ingest.py is standalone. Let's just call it.
            ingest_images(input_dir=DATA_DIR)
            
        st.success("Ingestion Complete!")
    else:
        st.warning("Please upload images first.")

# --- Query Section ---
# --- Query Section ---
st.header("2. Query Images")


if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "search_logs" not in st.session_state:
    st.session_state.search_logs = []

def start_search():
    st.session_state.search_running = True

with st.form("query_form"):
    query_text = st.text_input("Enter your query (e.g., 'a red car')")
    search_submitted = st.form_submit_button(
        "Search",
        disabled=st.session_state.search_running,
        on_click=start_search
    )

if search_submitted:
    if query_text:
        st.write(f"**Query:** {query_text}")
        
        # Clear previous results/logs
        st.session_state.search_results = None
        st.session_state.search_logs = []
        
        # Container for logs (temporary for this run)
        log_container = st.empty()
        
        def log_to_ui(message):
            st.session_state.search_logs.append(message)
            if verbose_mode:
                log_container.code("\n".join(st.session_state.search_logs))

        with st.spinner("Searching..."):
            try:
                results = rag_system.process_query(
                    query_text, 
                    log_callback=log_to_ui,
                    timeout=SEARCH_TIMEOUT_SEC
                )
                st.session_state.search_results = results
            finally:
                st.session_state.search_running = False
                st.rerun()
            
    else:
        st.warning("Please enter a query.")
        st.session_state.search_running = False

# Display Results (Persistent)
if st.session_state.search_results is not None:
    # Re-display logs if verbose
    if verbose_mode and st.session_state.search_logs:
        st.code("\n".join(st.session_state.search_logs))

    results = st.session_state.search_results
    if results:
        st.success(f"Found {len(results)} matches!")
        cols = st.columns(len(results))
        for idx, img_path in enumerate(results):
            with cols[idx]:
                try:
                    image = Image.open(img_path)
                    st.image(image, caption=f"Result {idx+1}", use_container_width=True)
                except Exception as e:
                    st.error(f"Could not load image {img_path}")
    else:
        st.warning("No matching objects found.")
