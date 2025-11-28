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
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def get_caption_collection(client):
    """
    Get or create the collection for image captions.
    """
    return client.get_or_create_collection(name=CAPTION_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


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


def get_collection_count(client):
    """
    Returns the number of items in the image collection.
    """
    collection = get_collection(client)
    return collection.count()


def clear_collection(client):
    """
    Deletes and recreates the collections to clear all data.
    """
    try:
        client.delete_collection(COLLECTION_NAME)
        client.delete_collection(CAPTION_COLLECTION_NAME)
    except ValueError:
        pass  # Collections might not exist
    
    # Recreate
    get_collection(client)
    get_caption_collection(client)


def manage_collection_limit(client, limit, new_count):
    """
    Ensures the collection does not exceed the limit by deleting the oldest items.
    """
    collection = get_collection(client)
    caption_collection = get_caption_collection(client)
    
    current_count = collection.count()
    total_count = current_count + new_count
    
    if total_count > limit:
        num_to_delete = total_count - limit
        print(f"Collection limit exceeded. Deleting {num_to_delete} oldest items...")
        
        # Get all metadata to find timestamps
        # Note: This might be slow for very large collections, but fine for ~1000
        result = collection.get(include=["metadatas"])
        ids = result["ids"]
        metadatas = result["metadatas"]
        
        # Create a list of (id, timestamp) tuples
        # Handle missing timestamps by assigning 0 (delete them first)
        items = []
        for i, meta in enumerate(metadatas):
            ts = meta.get("timestamp", 0) if meta else 0
            items.append((ids[i], ts))
            
        # Sort by timestamp (ascending = oldest first)
        items.sort(key=lambda x: x[1])
        
        # Select IDs to delete
        ids_to_delete = [item[0] for item in items[:num_to_delete]]
        
        # Delete from both collections
        collection.delete(ids=ids_to_delete)
        caption_collection.delete(ids=ids_to_delete)
        print(f"Deleted {len(ids_to_delete)} items.")
