import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "image_embeddings"
EMBEDDING_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"


def get_chroma_client():
    """
    Returns a persistent ChromaDB client so data is saved to disk
    and can be accessed by the query script.
    """
    # Using PersistentClient to save data to disk
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client


def get_collection(client):
    """
    Get or create the collection for image embeddings.
    """
    return client.get_or_create_collection(name=COLLECTION_NAME)


def get_embedding_model():
    """
    Load the MiniCLIP model for generating embeddings.
    """
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model
