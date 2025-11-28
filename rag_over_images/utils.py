import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration

# Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "image_embeddings"
EMBEDDING_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
CAPTION_COLLECTION_NAME = "image_captions"
TEXT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"


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


def get_caption_collection(client):
    """
    Get or create the collection for image captions.
    """
    return client.get_or_create_collection(name=CAPTION_COLLECTION_NAME)


def get_embedding_model():
    """
    Load the MiniCLIP model for generating embeddings.
    """
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def get_text_embedding_model():
    """
    Load the all-MiniLM-L6-v2 model for generating text embeddings for captions.
    """
    model = SentenceTransformer(TEXT_EMBEDDING_MODEL_NAME)
    return model


def get_caption_model():
    """
    Load the BLIP model for image captioning.
    """
    processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME)
    return processor, model
