import os
import sys
from PIL import Image
from tqdm import tqdm
import random

random.seed(42)
# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_chroma_client, get_collection, get_embedding_model

INPUT_DIR = "./data/train"
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

N_IMAGES = 1000


def ingest_images(input_dir=INPUT_DIR):
    """
    Reads images from input_dir, generates embeddings, and stores them in ChromaDB.
    """
    # Create input dir if it doesn't exist (for demo purposes)
    if not os.path.exists(input_dir):
        print(f"Creating directory: {input_dir}")
        os.makedirs(input_dir)
        print(f"Please put some images in {input_dir} and run this script again.")
        return

    # Get list of image files
    image_files = [
        f for f in os.listdir(input_dir) if f.lower().endswith(SUPPORTED_EXTS)
    ]

    # Randomly take N_IMAGES images

    image_files = random.sample(image_files, N_IMAGES)

    if not image_files:
        print(f"No images found in {input_dir}. Please add some images.")
        return

    print(f"Found {len(image_files)} images. Loading model...")

    # Initialize resources
    client = get_chroma_client()
    collection = get_collection(client)
    model = get_embedding_model()

    print("Model loaded. Starting ingestion...")

    ids = []
    embeddings = []
    metadatas = []

    for img_file in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(input_dir, img_file)

        try:
            # Load image
            image = Image.open(img_path)

            # Generate embedding
            # sentence-transformers encode supports PIL images
            embedding = model.encode(image).tolist()

            ids.append(img_file)
            embeddings.append(embedding)
            metadatas.append({"path": img_path})

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    if ids:
        print("Upserting to Vector DB...")
        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print(f"Successfully ingested {len(ids)} images.")
    else:
        print("No valid images processed.")


if __name__ == "__main__":
    ingest_images()
