import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm
import random

random.seed(42)
# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    get_chroma_client,
    get_collection,
    get_embedding_model,
    get_caption_model,
    get_text_embedding_model,
    get_caption_collection,
)

INPUT_DIR = "./data/train"
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

N_IMAGES = 1000


def ingest_images(input_dir=INPUT_DIR, n_images=N_IMAGES):
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

    # Randomly take n_images images, but not more than available
    n_images = min(n_images, len(image_files))
    image_files = random.sample(image_files, n_images)

    if not image_files:
        print(f"No images found in {input_dir}. Please add some images.")
        return

    print(f"Found {len(image_files)} images. Loading model...")

    # Initialize resources
    client = get_chroma_client()
    collection = get_collection(client)
    model = get_embedding_model()
    
    # Initialize captioning resources
    print("Loading captioning models...")
    caption_processor, caption_model = get_caption_model()
    text_embedding_model = get_text_embedding_model()
    caption_collection = get_caption_collection(client)

    print("Model loaded. Starting ingestion...")

    ids = []
    embeddings = []
    metadatas = []
    
    caption_ids = []
    caption_embeddings = []
    caption_metadatas = []

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

            # Generate caption
            inputs = caption_processor(image, return_tensors="pt")
            out = caption_model.generate(**inputs)
            caption = caption_processor.decode(out[0], skip_special_tokens=True)
            
            # Generate caption embedding
            caption_embedding = text_embedding_model.encode(caption).tolist()
            
            caption_ids.append(img_file)
            caption_embeddings.append(caption_embedding)
            caption_metadatas.append({"path": img_path, "caption": caption})

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    if ids:
        print("Upserting to Vector DB...")
        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print(f"Successfully ingested {len(ids)} images into image collection.")
        
        print("Upserting to Caption DB...")
        caption_collection.upsert(
            ids=caption_ids,
            embeddings=caption_embeddings,
            metadatas=caption_metadatas
        )
        print(f"Successfully ingested {len(caption_ids)} captions.")
    else:
        print("No valid images processed.")


def main():
    parser = argparse.ArgumentParser(description="Ingest images into vector database")
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=1000,
        help="Maximum number of images to ingest",
    )
    args = parser.parse_args()
    ingest_images(args.input_dir, args.max_images)


if __name__ == "__main__":
    main()
