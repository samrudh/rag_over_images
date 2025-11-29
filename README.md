# RAG over Images

A clever computer vision system that helps you find specific objects in your photo collection using natural language queries. Think of it as having a smart assistant that can locate "the red car" or "a store sign" in hundreds of images without you manually searching through them. Perfect for organizing personal photos or exploring datasets in a fun, interactive way!

## Key Features
- **Semantic Query Caching**: Instantly returns results for similar past queries (e.g., "red car" hits cache for "a red car").
- **Smart Load Management**: Search button automatically disables during processing to prevent database overload and double submissions.
- **Observability**: "Verbose Mode" toggle in the UI provides real-time logs of the RAG pipeline (embedding, retrieval, grounding).
- **Smart Search based on public LLM**: Generates intelligent query suggestions using Gemini for enhanced search discovery.
- **LLM based Validation**: Uses Gemini to verify if the search results actually contain the requested object, providing a second layer of intelligent filtering.
- **Graceful Timeouts**: Queries exceeding 10 seconds are handled gracefully to ensure system responsiveness.

## Data Flow Documentation

### Ingestion Flow

The ingestion process processes images, generates embeddings and captions, and stores them in ChromaDB for retrieval.

**Step-by-step Process:**

1. **Input Collection**:
   - Scan the specified input directory (`./data/train` or uploaded files) for supported image formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`)
   - Randomly sample up to `n_images` (default 100 for train mode, up to 200 for uploads) if more images are available

2. **Model Initialization**:
   - Initialize ChromaDB persistent client (stored in `./chroma_db`)
   - Load CLIP ViT-B-32 model for image embeddings (`sentence-transformers/clip-ViT-B-32`)
   - Load BLIP model for image captioning (`Salesforce/blip-image-captioning-base`)
   - Load All-MiniLM-L6-v2 model for caption text embeddings (`sentence-transformers/all-MiniLM-L6-v2`)

3. **Mode Handling**:
   - **Clean mode**: Delete existing `image_embeddings` and `image_captions` collections
   - **Append mode**: Check current collection size against limit (default 1000), delete oldest items (by timestamp) if limit would be exceeded

4. **Per-Image Processing Loop**:
   - Load the image using PIL
   - Generate CLIP visual embedding for the image (normalized)
   - Use BLIP to generate a descriptive caption for the image
   - Generate All-MiniLM text embedding for the caption (normalized)

5. **Data Storage**:
   - **Image Collection (`image_embeddings`)**: Store image embeddings with metadata (image path, processing timestamp)
   - **Caption Collection (`image_captions`)**: Store caption embeddings with metadata (image path, generated caption, processing timestamp)
   - Use ChromaDB's batch upsert for efficiency

6. **Smart Query Suggestions (Optional)**:
   - If enabled with Gemini API key: Fetch all captions from DB, use Gemini-2.5-flash to generate 50 diverse query suggestions, save to `data/suggested_queries.json`

```
    +-------------+    +-----------------+    +------------------+    +-------------+
    |   Images    | -> | Model Loading  | -> | Embedding &      | -> |  ChromaDB   |
    |             |    |   & Client     |    | Caption Gen      |    |  Storage    |
    +-------------+    +-----------------+    +------------------+    +-------------+
```

### Query Flow

The query process retrieves relevant images using multi-modal similarity and performs object detection for precise matching.

**Step-by-step Process:**

1. **Query Preparation**:
   - Generate CLIP visual embedding for the text query (treating query as image-like)
   - Generate All-MiniLM text embedding for the text query
   - Check query cache for semantically similar previous queries (using cosine similarity > 0.85 threshold)

2. **Retrieval Phase**:
   - **Visual Search**: Query the image collection for top 10 visual matches using CLIP embedding, filter results by visual similarity threshold (default 0.25)
   - **Caption Search**: Query the caption collection for top 10 matches using text embedding, filter results by caption similarity threshold (default 0.25)
   - Merge results from both searches, removing duplicates, tracking source (Visual/Caption/Both)

3. **Grounding Phase (Florence-2 Object Detection)**:
   - For each candidate image: Use Florence-2-base-ft model to perform phrase grounding (detect objects in image matching the query text)
   - Generate bounding boxes for detected objects using the `<CAPTION_TO_PHRASE_GROUNDING>` task
   - Draw green bounding boxes and labels on detected objects, save visualization to output directory

4. **Result Compilation**:
   - Return paths to output images with bounding boxes, marked by matched sources (for logging/attribution)
   - Cache the query embedding and results to avoid re-processing similar queries
   - Apply timeout (default 10 seconds) to prevent long-running queries

5. **Smart Validation (Optional)**:
   - If enabled, sends the query and retrieved images to Gemini-2.5-flash
   - The LLM analyzes the images to confirm if the requested object is truly present
   - Returns a "Yes/No" validation with a brief explanation, displayed to the user

6. **Caching Mechanism**:
   - Maintains a deque cache of 10 recent queries
   - Uses cosine similarity to find cached results for semantically close queries
   - Avoids re-running expensive grounding on repeated or similar searches

```
    +-------------+    +-----------------+    +------------------+    +-------------+    +--------------+
    |   Query     | -> |  Embedding     | -> |   Retrieval      | -> |  Grounding  | -> |  LLM Validation  |
    |             |    |   Generation   |    | (Visual+Caption) |    | & Results   |    |  (Optional)  |
    +-------------+    +-----------------+    +------------------+    +-------------+    +--------------+
```

### Models Used

| Detail              | Value |
|---------------------|-------|
| Model Name          | sentence-transformers/clip-ViT-B-32 |
| Model Context Limit | Not strictly applicable (multimodal); Image resolution: typically 224x224 pixels. Text sequence length: typically ≤77 tokens. |
| Model Parameters    | Approximately 151 million parameters. |
| Key Advantages      | Creates joint image and text embeddings in a shared vector space, which is essential for cross-modal tasks like image-text retrieval. It enables zero-shot image classification, allowing the model to classify images into categories it was not explicitly trained on, using natural language prompts. |

| Detail              | Value |
|---------------------|-------|
| Model Name          | Salesforce/blip-image-captioning-base |
| Model Context Limit | Input image resolution: typically 384x384 or 224x224. Output text length: typically ≤20 tokens (can be controlled during generation). |
| Model Parameters    | Approximately 109 million parameters. |
| Key Advantages      | State-of-the-art performance for generating accurate and contextually relevant captions for images. Its unified vision-language pre-training framework allows it to flexibly transfer to both understanding and generation tasks. |

| Detail              | Value |
|---------------------|-------|
| Model Name          | sentence-transformers/all-MiniLM-L6-v2 |
| Model Context Limit | Max input length is 256 word pieces (longer text is truncated). |
| Model Parameters    | Approximately 22.7 million parameters. |
| Key Advantages      | Extremely compact and fast model that generates high-quality, 384-dimensional dense vector embeddings for sentences and short paragraphs. This makes it ideal for efficient deployment in applications like semantic search and clustering. |

## Installation Instructions

### Prerequisites
- Python 3.10 or higher
- uv package manager (recommended) or pip

### Quick Setup with uv (Recommended)
```bash
# Clone the repository
git clone https://github.com/samrudh/rag_over_images.git
cd rag_over_images

# Install with uv (fast dependency resolution)
uv sync

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Alternative Setup with pip
```bash
# Install dependencies
pip install -r requirements.txt

# Or install the package
pip install -e .
```

## System Configuration

### Database Configuration
- The system maintains a FIFO limit of 100 images maximum in the database. When appending new images that would exceed this limit, the oldest images are automatically deleted to make space.
- Results are cached in memory for up to 10 recent queries to improve performance for similar searches.

### UI Configuration
- **Visual Similarity Control**: Slider adjusts the minimum similarity score (0.0-1.0) for matches from CLIP-based visual search. Lower values allow more matches, higher values are stricter.
- **Caption Similarity Control**: Slider adjusts the minimum similarity score for text caption matches. Controls how closely captions must match the query.

### System Requirements
- First startup will download approximately 4GB of model files (Florence-2, CLIP, BLIP), which may take several minutes on slower connections.
- No environment variables are required - all configuration is handled through the UI or command line parameters.

## References

### Core Technologies
- [ChromaDB](https://www.trychroma.com/) - Vector database for embeddings
- [Sentence Transformers](https://www.sbert.net/) - CLIP and other embedding models
- [Florence-2](https://github.com/microsoft/Florence-2) - Microsoft's vision foundation model
- [Streamlit](https://streamlit.io/) - Web application framework

### Datasets
- [PASCAL VOC 2012](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset) - Object detection dataset


### Research Papers
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Florence-2: Advancing Multimodal Understanding with Universal VLM](https://arxiv.org/abs/2311.06242)

## Additional Notes

### Usage Examples

#### Command Line Interface
```bash
# Ingest images into the vector database
rag-ingest --input-dir ./data/train --max-images 1000

# Query for objects in images (single query mode)
rag-query --query "a black sedan car"

# Run interactive query mode
rag-query --interactive

# Start the web application
rag-app
```

#### Alternative  to run web application
```bash
streamlit run rag_over_images/app.py
```

#### Web Interface
1. Start the application: `rag-app`
   - Or run directly: `venv/bin/streamlit run rag_over_images/app.py`
2. Open browser to `http://localhost:8501`
3. Upload images or use existing dataset
4. Ask questions like "Is there a bicycle?" or "Find all dogs"

### Project Structure
```
rag_over_images/
├── pyproject.toml          # Modern Python package configuration
├── rag_over_images/        # Main package
│   ├── __init__.py        # Package initialization
│   ├── app.py            # Streamlit web interface
│   ├── utils.py          # Shared utilities
│   ├── ingestion/        # Image processing module
│   └── query/            # Query and retrieval module
├── data/                 # Image datasets (add your images here)
├── outputs/              # Generated results and cache
└── chroma_db/           # Vector database storage
```

### Performance Tips
- **Batch Processing**: Use `--max-images` flag to limit initial ingestion
- **GPU Acceleration**: Install CUDA-compatible PyTorch for faster embedding generation
- **Memory Management**: Large datasets may require increasing system memory limits
- **Database Optimization**: ChromaDB automatically optimizes queries, but consider indexes for very large collections

### Troubleshooting
- **Import Errors**: Ensure virtual environment is activated
- **Memory Issues**: Reduce batch size or use smaller embedding models
- **Model Downloads**: First run may take time to download Florence-2 and CLIP models
- **Database Corruption**: Delete `chroma_db/` directory and re-ingest images

### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with clear description

### License
This project is licensed under the MIT License - see the LICENSE file for details.
