# RAG over Images

A clever computer vision system that helps you find specific objects in your photo collection using natural language queries. Think of it as having a smart assistant that can locate "the red car" or "a store sign" in hundreds of images without you manually searching through them. Perfect for organizing personal photos or exploring datasets in a fun, interactive way!

## Installation Instructions

### Prerequisites
- Python 3.8 or higher
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

## Environment Assumptions

### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python Version**: 3.8+
- **Memory**: 8GB+ recommended (embedding models are memory-intensive)
- **Storage**: 2GB+ free space for models and image cache

### External Dependencies
- **ChromaDB**: Vector database for image embeddings
- **Sentence Transformers**: For image embeddings (CLIP-based)
- **Florence-2**: Microsoft's vision-language model for object detection
- **Streamlit**: For web interface (optional)
- **Torch/PyTorch**: Deep learning framework

### Optional Components
- **GPU**: CUDA-capable GPU for faster processing (recommended)
- **Docker**: For containerized deployment

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

# Query for objects in images
rag-query "a black sedan car"

# Start the web application
rag-app
```

#### Web Interface
1. Start the application: `rag-app`
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