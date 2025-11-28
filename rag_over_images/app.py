import os
import subprocess
import streamlit as st
from PIL import Image
import shutil
import time

# Import logic
from utils import get_chroma_client, get_collection, get_embedding_model, get_collection_count
from query.query import RAGQuerySystem
from ingestion.ingest import ingest_images, SUPPORTED_EXTS

# Constants
DATA_DIR = "./data/train"
UPLOAD_DIR = "./data/uploaded"
OUTPUT_DIR = "./outputs"
SEARCH_TIMEOUT_SEC = 10  # Timeout for search button disable

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
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
    st.session_state["resources_loaded"] = True
st.sidebar.success("System Ready!")

# Database Status Visualization (Battery/Jar)
st.sidebar.markdown("---")
st.sidebar.header("Database Status")
try:
    client = get_chroma_client()
    count = get_collection_count(client)
    limit = 100
    percentage = min(count / limit, 1.0)
    
    # Battery/Jar Visual
    st.sidebar.progress(percentage)
    st.sidebar.caption(f"{count} / {limit} images ({int(percentage*100)}%)")
    
    if count >= limit:
        st.sidebar.warning("Database Full (FIFO active)")
except Exception as e:
    st.sidebar.error("Could not connect to DB")

# Verbose Toggle
verbose_mode = st.sidebar.toggle(
    "Observability Control", 
    value=False,
    disabled=st.session_state.search_running
)


# --- Ingestion Section ---
st.header("1. Ingest Images")
uploaded_files = st.file_uploader(
    "Upload images to ingest", accept_multiple_files=True, type=["jpg", "jpeg", "png"]
)

# Ingestion Buttons
col1, col2 = st.columns(2)

with col1:
    clean_ingest = st.button("Clean & Ingest", type="primary", help="Wipes DB and ingests new images")
with col2:
    append_ingest = st.button("Append", help="Adds to DB (deletes oldest if full)")
    
if clean_ingest or append_ingest:
    mode = "clean" if clean_ingest else "append"
    
    target_dir = DATA_DIR
    num_images_to_ingest = 100
    
    if uploaded_files:
        # User Upload Mode
        if len(uploaded_files) > 200:
            st.error("Please upload a maximum of 200 images.")
            st.stop()
            
        # Clear upload directory to ensure isolation
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR)
        
        # Save uploaded files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        st.info(f"Saved {len(uploaded_files)} images to {UPLOAD_DIR}")
        target_dir = UPLOAD_DIR
        num_images_to_ingest = len(uploaded_files)
    else:
        # Default Mode
        st.info(f"No files uploaded. Using default dataset from {DATA_DIR} (limit 100).")

    # Run Ingestion
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(current, total, message):
        progress = min(current / total, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"{message} ({current}/{total})")

    with st.spinner(f"Ingesting images ({mode} mode)..."):
        ingest_images(
            input_dir=target_dir, 
            n_images=num_images_to_ingest, 
            progress_callback=update_progress,
            mode=mode
        )
        
    progress_bar.progress(1.0)
    status_text.text("Ingestion Complete!")
    st.success("Ingestion Complete!")
    
    # Clear cache to ensure RAG system reloads with new collection
    st.cache_resource.clear()
    
    # Rerun to update sidebar status
    time.sleep(1)
    st.rerun()

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
    
    # Similarity Threshold Sliders
    st.markdown("### Search Configuration")
    col1, col2 = st.columns(2)
    with col1:
        visual_threshold = st.slider("Visual Similarity Control", 0.0, 1.0, 0.6, 0.05, help="Minimum similarity for VLM matches")
    with col2:
        caption_threshold = st.slider("Caption Similarity Control", 0.0, 1.0, 0.6, 0.05, help="Minimum similarity for Caption matches")

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
                    timeout=SEARCH_TIMEOUT_SEC,
                    visual_threshold=visual_threshold,
                    caption_threshold=caption_threshold
                )
                st.session_state.search_results = results
            except Exception as e:
                st.error(f"An error occurred during search: {e}")
                # Don't rerun immediately on error so user can see it
            finally:
                st.session_state.search_running = False
                if st.session_state.search_results is not None:
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
        for idx, item in enumerate(results):
            with cols[idx]:
                try:
                    img_path = item["path"]
                    source = item["source"]
                    image = Image.open(img_path)
                    st.image(image, caption=f"Result {idx+1} ({source})", width="stretch")
                except Exception as e:
                    st.error(f"Could not load image {img_path}")
    else:
        st.warning("No matching objects found.")
