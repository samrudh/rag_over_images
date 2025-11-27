import os
import subprocess
import streamlit as st
from PIL import Image
import shutil

# Import logic
from utils import get_chroma_client, get_collection, get_embedding_model
from query.query import load_florence_model, run_grounding, draw_boxes
from ingestion.ingest import ingest_images, SUPPORTED_EXTS

# Constants
DATA_DIR = "./data/train"
OUTPUT_DIR = "./outputs"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Page Config
st.set_page_config(page_title="RAG over Images", layout="wide")
st.title("🖼️ RAG over Images")

# Sidebar for Resource Loading
st.sidebar.header("System Status")
if "resources_loaded" not in st.session_state:
    st.session_state["resources_loaded"] = False


@st.cache_resource
def load_resources():
    st.sidebar.text("Loading Embedding Model...")
    embed_model = get_embedding_model()
    st.sidebar.text("Loading ChromaDB...")
    client = get_chroma_client()
    collection = get_collection(client)
    st.sidebar.text("Loading Florence-2...")
    florence_model, florence_processor = load_florence_model()
    return embed_model, client, collection, florence_model, florence_processor


# Load resources
try:
    embed_model, client, collection, florence_model, florence_processor = (
        load_resources()
    )
    st.sidebar.success("All Models Loaded!")
    st.session_state["resources_loaded"] = True
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["📤 Ingestion", "🔍 Query"])

# --- Ingestion Tab ---
with tab1:
    st.header("Upload Images")
    uploaded_files = st.file_uploader(
        "Choose images", accept_multiple_files=True, type=["png", "jpg", "jpeg"]
    )

    if st.button("Ingest Uploaded Images"):
        if uploaded_files:
            # Save uploaded files to DATA_DIR
            saved_count = 0
            for uploaded_file in uploaded_files:
                # Simple file save
                with open(os.path.join(DATA_DIR, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_count += 1
            st.success(f"Saved {saved_count} images to {DATA_DIR}")

            # Run ingestion
            with st.spinner("Embedding and Indexing..."):
                ingest_images(input_dir=DATA_DIR)
            st.success("Ingestion Complete!")
        else:
            st.warning("Please upload some files first.")

    # Show current index stats
    st.divider()
    st.subheader("Current Index Stats")
    try:
        count = collection.count()
        st.info(f"Total Images in Vector DB: {count}")
    except:
        st.text("DB not initialized yet.")

# --- Query Tab ---
with tab2:
    st.header("Search & Grounding")

    query = st.text_input(
        "Ask a question about your images:", placeholder="e.g. Find the red car"
    )

    if st.button("Search") and query:
        with st.spinner("Retrieving Candidates..."):
            # 1. Retrieval
            query_emb = embed_model.encode(query).tolist()
            results = collection.query(
                query_embeddings=[query_emb], n_results=3  # Top-3
            )

            cand_paths = results["metadatas"][0] if results["metadatas"] else []
            cand_paths = [m["path"] for m in cand_paths]

            if not cand_paths:
                st.warning("No images found in index.")
            else:
                st.subheader("Top Retrieved Images & Phrase Grounding Results")

                cols = st.columns(3)

                for idx, img_path in enumerate(cand_paths):
                    with cols[idx]:
                        st.text(f"Candidate {idx+1}")

                        # 2. Grounding
                        try:
                            result = run_grounding(
                                florence_model, florence_processor, img_path, query
                            )

                            # Draw boxes
                            output_filename = os.path.join(
                                OUTPUT_DIR, f"result_{idx}.jpg"
                            )
                            # We need to ensure we don't overwrite per query if concurrent, but for local single user it's fine.
                            # Just use unique name or overwrite.

                            saved_path = draw_boxes(img_path, result, output_filename)

                            if saved_path:
                                st.image(
                                    saved_path,
                                    caption=f"Match Found!",
                                    use_container_width=True,
                                )
                            else:
                                st.image(
                                    img_path,
                                    caption="No specific object match",
                                    use_container_width=True,
                                )

                        except Exception as e:
                            st.error(f"Error processing {img_path}: {e}")


def main():
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
