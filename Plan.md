# 🖼️ RAG over Images Implementation Plan

## 1. Goal

Build a system that allows users to upload multiple images, ask natural language questions about objects in those images (e.g., "is there a red car?"), and receive an answer with the object highlighted by a bounding box.

## 2. Key Constraints

- **Locally Hostable/Efficient**: Designed for small-scale deployment (e.g., Runpod, local GPU).
- **No "Big" LLMs/VLMs**: Prioritize efficient, small-scale models (SLMs).
- **Memory Data Store**: Use ephemeral, in-memory vector stores.
- **Single Object Query**: Assume queries target a specific object description.
- **Architecture**: Python-based with a simple REST/Web interface.

## 3. Architecture Overview: Retrieval-Augmented Grounding

The solution utilizes a two-stage pipeline to handle many images efficiently:

1.  **Retrieval (The "RAG" part)**: Use CLIP embeddings to quickly filter and select the Top-K (e.g., $K=3$) images most likely to contain the answer. This avoids running the expensive VLM on all images.
2.  **Grounding/Detection**: Use a specialized, small Vision-Language Model (VLM) like Florence-2 to perform text-based object detection (phrase grounding) on only the selected subset, explicitly locating and highlighting the object.

## 4. Technology Stack Choices

### 4.1. Vision/Grounding Model (Bounding Box Detection)

This model runs on the retrieved images to perform the final object localization.

| Choice | Pros | Cons |
| :--- | :--- | :--- |
| **1. Microsoft Florence-2-base** | Multitasking VLM (Grounding, Captioning, Detection); Very small (0.23B parameters); Fast inference on GPU. | Needs specific finetuning or task prompts; Detection accuracy is lower than dedicated SOTA detection models. |
| **2. OWL-ViT (Open-Vocabulary)** | Dedicated Open-Vocabulary Detection; Specifically built for natural language object localization. | Slightly larger than Florence-2-base, and not a general VLM (can't perform captioning). |
| **3. Grounding DINO + SAM** | State-of-the-Art accuracy for zero-shot grounding (DINO) and segmentation (SAM). | Slower inference; Requires running two large models sequentially; Higher VRAM usage. |

**Recommended**: **Florence-2-base** for its efficiency, single-model solution, and multitasking capability (useful for generating retrieval summaries).

### 4.2. Image Embedding Model (for Retrieval)

This model translates the image/text into a vector space for semantic search.

| Choice | Pros | Cons |
| :--- | :--- | :--- |
| **1. CLIP ViT-B/32** | Industry Standard; Excellent zero-shot performance for similarity search; Small model size (~400MB). | Shorter context length may sometimes miss fine-grained details compared to larger versions. |
| **2. MiniCLIP / SigLIP** | Much smaller and faster inference (very low latency) for extremely rapid retrieval. | Slightly lower accuracy on complex, abstract or nuanced visual/semantic queries. |
| **3. CLIP ViT-L/14** | Highest Accuracy; Captures more complex visual and semantic relationships for better relevance. | Larger size (~1.5GB) requires more VRAM and increases load time. |

**Recommended**: **CLIP ViT-B/32** (`sentence-transformers/clip-ViT-B-32`) for its excellent balance of performance and efficiency.

### 4.3. Vector Database (In-Memory)

Used to store embeddings and quickly retrieve the most relevant images.

| Choice | Pros | Cons |
| :--- | :--- | :--- |
| **1. ChromaDB** | Python Native, Easy Setup; Simple API; No external dependencies; Ideal for in-memory use. | Performance may be slightly slower than FAISS for extremely large, high-throughput indexes. |
| **2. FAISS** | Extremely Fast Retrieval; Highly optimized C++ backend; Best for high-dimensional vectors. | Setup is slightly more complex (requires C++ binaries/libraries); Primarily a numerical index, not a full database. |
| **3. Annoy** | Efficient memory usage; Good for static indexes where updates are rare; Created by Spotify. | Slower build/write time than FAISS; Generally best for read-only scenarios. |

**Recommended**: **ChromaDB** for its simplicity, ease of use in a local Python environment, and zero external dependency requirement.

## 5. Ingestion Flow (Index Creation)

The ingestion process prepares images for fast retrieval.

1.  **Image Upload**: User uploads $N$ images.
2.  **Vector Embedding (CLIP)**:
    -   Load CLIP ViT-B/32.
    -   For each image: Generate a global image embedding vector.
    -   Store the vector in ChromaDB with the file path and image dimensions as metadata.
3.  **Hybrid Search Enrichment (Optional but recommended)**:
    -   Load Florence-2-base to generate a short caption for the image (e.g., "A yellow taxi on a city street.").
    -   Embed this caption using the CLIP text encoder.
    -   Store this text vector in the same index alongside the image vector to enable Hybrid Search (querying for either visual or textual similarity).

**Outcome**: A searchable, in-memory index of image and text vectors.

## 6. Query Flow (Search and Grounding)

The flow when a user asks: *"Find the red sedan."*

1.  **Query Embedding**: Convert user query text into a vector using the CLIP text encoder.
2.  **Retrieval**: Query ChromaDB using the text vector to find the Top-K ($K=3$) most relevant images/metadata.
3.  **Object Detection & Grounding**:
    -   Load Florence-2-base for text-guided detection.
    -   Iterate through the $K$ retrieved images.
    -   Run the model using the grounding task prompt: `<CAPTION_TO_PHRASE_GROUNDING>` with the user query.
4.  **Response Generation**:
    -   **Parse Output**: Extract bounding boxes and confidence scores.
    -   **Filtering**: Discard bounding boxes with confidence below a threshold (e.g., 0.75).
    -   **Validation**: If valid boxes are found, the answer is "Yes, the object is present."
    -   **Visualization**: Use Pillow or OpenCV to draw the bounding boxes on the original image.
5.  **Final Output**: Return the annotated image(s) and a text confirmation.

## 7. Development Roadmap

- [ ] **Setup Environment**: Install `torch`, `transformers`, `chromadb`, `Pillow`, `tqdm`.
- [ ] **Ingestion Script** (`ingest.py`): Implement image loading, CLIP embedding, optional Florence-2 captioning, and ChromaDB indexing.
- [ ] **Retrieval & Grounding Script** (`query.py`): Implement query embedding, retrieval, Florence-2 grounding, and visualization logic.
- [ ] **API/UI**: Wrap the core logic in a simple Streamlit or Gradio interface for demo purposes.
- [ ] **Deployment**: Prepare `requirements.txt` and deployment notes (e.g., for Runpod or Modal Labs).
