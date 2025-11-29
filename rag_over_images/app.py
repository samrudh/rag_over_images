import os
import subprocess
import streamlit as st
from PIL import Image
import shutil
import time

# Import logic
from utils import (
    get_chroma_client,
    get_collection,
    get_embedding_model,
    get_collection_count,
)
from query.query import RAGQuerySystem
from ingestion.ingest import ingest_images, SUPPORTED_EXTS
import LLM
import json
import random

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

if "system_logs" not in st.session_state:
    st.session_state.system_logs = []


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
    "Observability Control", value=False, disabled=st.session_state.search_running
)

if verbose_mode:
    st.sidebar.markdown("### System Logs")
    log_container = st.sidebar.empty()
    if st.session_state.system_logs:
        # Show last 50 lines to keep it clean
        log_container.code("\n".join(st.session_state.system_logs[-50:]))
    else:
        log_container.info("No logs yet.")
else:
    # Use st.empty() as a placeholder instead of None to satisfy type checker
    log_container = st.empty()

# Smart Query Suggestions Control
st.sidebar.markdown("---")
st.sidebar.header("Smart Features")
use_smart_suggestions = st.sidebar.toggle("Enable Smart Query Suggestions", value=False)
gemini_api_key = ""
if use_smart_suggestions:
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if not gemini_api_key:
        st.sidebar.warning("API Key required for suggestions")


# --- Ingestion Section ---
st.header("1. Ingest Images")
uploaded_files = st.file_uploader(
    "Upload images to ingest", accept_multiple_files=True, type=["jpg", "jpeg", "png"]
)

# Ingestion Buttons
col1, col2 = st.columns(2)

with col1:
    clean_ingest = st.button(
        "Clean & Ingest", type="primary", help="Wipes DB and ingests new images"
    )
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
        st.info(
            f"No files uploaded. Using default dataset from {DATA_DIR} (limit 100)."
        )

    # Run Ingestion
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Run Ingestion
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(current: int, total: int, message: str) -> None:
        progress = min(current / total, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing... ({current}/{total})")

        # Add to logs
        st.session_state.system_logs.append(message)
        if verbose_mode and log_container:
            log_container.code("\n".join(st.session_state.system_logs[-50:]))

    with st.spinner(f"Ingesting images ({mode} mode)..."):
        ingest_images(
            input_dir=target_dir,
            n_images=num_images_to_ingest,
            progress_callback=update_progress,
            mode=mode,
        )

    progress_bar.progress(1.0)
    status_text.text("Ingestion Complete!")
    st.success("Ingestion Complete!")

    # Generate Smart Suggestions if enabled
    if use_smart_suggestions and gemini_api_key:
        with st.spinner("Generating smart query suggestions..."):
            st.session_state.system_logs.append("\n--- Smart Query Suggestions ---")
            st.session_state.system_logs.append(
                "LLM smart query suggestion process start"
            )
            if verbose_mode and log_container:
                log_container.code("\n".join(st.session_state.system_logs[-50:]))

            try:
                # Fetch captions from DB
                client = get_chroma_client()
                caption_collection = client.get_collection("image_captions")
                # Get all captions (limit to reasonable amount if needed, but here we take all)
                results = caption_collection.get(include=["metadatas"])
                captions = [
                    m["caption"] for m in results["metadatas"] if "caption" in m
                ]

                if captions:
                    suggestions = LLM.generate_query_suggestions(
                        captions, gemini_api_key
                    )
                    if suggestions:
                        # Save to file
                        with open("data/suggested_queries.json", "w", encoding="utf-8") as f_out:
                            json.dump(suggestions, f_out)

                        st.session_state.system_logs.append(
                            f"LLM smart query suggestion process complete."
                        )
                        st.session_state.system_logs.append(
                            f"Generated {len(suggestions)} suggestions."
                        )
                        st.session_state.system_logs.append(
                            f"Examples: {', '.join(suggestions[:3])}..."
                        )
                        if verbose_mode and log_container:
                            log_container.code(
                                "\n".join(st.session_state.system_logs[-50:])
                            )

                        st.success(f"Generated {len(suggestions)} smart suggestions!")
                else:
                    st.warning("No captions found to generate suggestions.")
            except Exception as e:
                st.error(f"Failed to generate suggestions: {e}")

    # Clear cache to ensure RAG system reloads with new collection
    st.cache_resource.clear()

    # Rerun to update sidebar status
    time.sleep(1)
    st.rerun()

# --- Query Section ---
st.header("2. Query Images")


if "search_results" not in st.session_state:
    st.session_state.search_results = None


def start_search():
    st.session_state.search_running = True


with st.form("query_form"):
    # Load and display suggestions
    suggestions_file = "data/suggested_queries.json"
    if os.path.exists(suggestions_file):
        try:
            with open(suggestions_file, "r", encoding="utf-8") as f_in:
                all_suggestions = json.load(f_in)
            if all_suggestions:
                # Pick 3-4 random suggestions
                random_suggestions = random.sample(
                    all_suggestions, min(4, len(all_suggestions))
                )
                st.markdown("**Try these queries:**")

                # Create a string of pills or just text
                # Streamlit pills are new, but let's use markdown for compatibility
                suggestion_text = " | ".join([f"`{s}`" for s in random_suggestions])
                st.markdown(suggestion_text)
        except Exception as e:
            pass  # Ignore errors in loading suggestions

    query_text = st.text_input("Enter your query (e.g., 'a red car')")

    # Similarity Threshold Sliders
    st.markdown("### Search Configuration")
    col1, col2 = st.columns(2)
    with col1:
        visual_threshold = st.slider(
            "Visual Similarity Control",
            0.0,
            1.0,
            0.255,
            0.05,
            help="Minimum similarity for VLM matches",
        )
    with col2:
        caption_threshold = st.slider(
            "Caption Similarity Control",
            0.0,
            1.0,
            0.255,
            0.05,
            help="Minimum similarity for Caption matches",
        )

    # Validation Toggle
    use_validation = False
    if gemini_api_key:
        use_validation = st.checkbox("LLM based validation using Gemini", value=False)

    search_submitted = st.form_submit_button(
        "Search", disabled=st.session_state.search_running, on_click=start_search
    )

if search_submitted:
    if query_text:
        st.write(f"**Query:** {query_text}")

        # Clear previous results/logs
        st.session_state.search_results = None
        st.session_state.search_logs = []

        # Container for logs (temporary for this run)
        # log_container = st.empty() # Removed in favor of sidebar

        def log_to_ui(message: str) -> None:
            st.session_state.system_logs.append(message)
            if verbose_mode and log_container:
                log_container.code("\n".join(st.session_state.system_logs[-50:]))

        with st.spinner("Searching..."):
            try:
                results = rag_system.process_query(
                    query_text,
                    log_callback=log_to_ui,
                    timeout=SEARCH_TIMEOUT_SEC,
                    visual_threshold=visual_threshold,
                    caption_threshold=caption_threshold,
                )
                st.session_state.search_results = results
                
                # Smart Validation
                if use_validation and results and gemini_api_key:
                    with st.spinner("Validating results with Gemini..."):
                        image_paths = [r["path"] for r in results]
                        validation_response, valid_indices = LLM.validate_search_results(
                            query_text, image_paths, gemini_api_key
                        )
                        st.session_state.validation_response = validation_response
                        
                        # Filter results based on valid_indices
                        if valid_indices is not None:
                            filtered_results = [results[i] for i in valid_indices if i < len(results)]
                            if len(filtered_results) < len(results):
                                log_to_ui(f"Smart Validation filtered out {len(results) - len(filtered_results)} images.")
                            st.session_state.search_results = filtered_results
                        else:
                             st.session_state.search_results = results

                else:
                    st.session_state.validation_response = None
                    
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
    # Re-display logs if verbose - ALREADY IN SIDEBAR
    # if verbose_mode and st.session_state.search_logs:
    #     st.code("\n".join(st.session_state.search_logs))

    results = st.session_state.search_results
    
    # Display Validation Response if available
    if "validation_response" in st.session_state and st.session_state.validation_response:
        st.info(f"**LLM Validation:**\n\n{st.session_state.validation_response}")

    if results:
        st.success(f"Found {len(results)} matches!")
        cols = st.columns(len(results))
        for idx, item in enumerate(results):
            with cols[idx]:
                try:
                    img_path = item["path"]
                    source = item["source"]
                    image = Image.open(img_path)
                    st.image(
                        image, caption=f"Result {idx+1} ({source})", width="stretch"
                    )
                except Exception as e:
                    st.error(f"Could not load image {img_path}")
    else:
        st.warning("No matching objects found.")
