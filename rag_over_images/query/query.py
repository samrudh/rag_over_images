import os
import sys
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from collections import deque
from scipy.spatial.distance import cosine
import time

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    get_chroma_client,
    get_collection,
    get_embedding_model,
    get_text_embedding_model,
    get_caption_collection,
)

FLORENCE_MODEL_ID = "microsoft/Florence-2-base-ft"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def load_florence_model():
    try:
        print(f"Loading Florence-2 model ({FLORENCE_MODEL_ID})...")
        # trust_remote_code=True is required for Florence-2
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = AutoModelForCausalLM.from_pretrained(
                FLORENCE_MODEL_ID,
                trust_remote_code=True,
            ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True)
        return model, processor
    except Exception as e:
        print(f"Warning: Failed to load Florence-2 model: {e}")
        print("Grounding will be disabled.")
        return None, None


def run_grounding(model, processor, image_path, text_query):
    """
    Runs Phrase Grounding on the image for the given text query.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Task prompt for Florence-2 Phrase Grounding
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    prompt = task_prompt + text_query

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        DEVICE, torch.float32
    )

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Parse the output
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )

    return parsed_answer


def draw_boxes(image_path, parsed_result, output_path):
    """
    Draws bounding boxes on the image and saves it.
    """
    if not parsed_result or "<CAPTION_TO_PHRASE_GROUNDING>" not in parsed_result:
        return None

    data = parsed_result["<CAPTION_TO_PHRASE_GROUNDING>"]
    # Start with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        return None

    # data structure: {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['label', ...]}
    bboxes = data.get("bboxes", [])
    labels = data.get("labels", [])

    found = False
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
        found = True

    if found:
        cv2.imwrite(output_path, img)
        return output_path
    return None


class QueryCache:
    def __init__(self, maxlen=10, similarity_threshold=0.85):
        self.cache = deque(maxlen=maxlen)
        self.threshold = similarity_threshold

    def find_similar(self, query_embedding):
        """
        Finds a cached result if the query is semantically similar.
        Returns the cached result if found, else None.
        """
        for cached_emb, cached_result in self.cache:
            # Calculate cosine similarity
            # 1 - cosine distance = cosine similarity
            similarity = 1 - cosine(query_embedding, cached_emb)
            if similarity >= self.threshold:
                return cached_result
        return None

    def add(self, query_embedding, result):
        """Adds a query and its result to the cache."""
        self.cache.append((query_embedding, result))


class RAGQuerySystem:
    def __init__(self):
        print("Initializing Query System...")
        self.client = get_chroma_client()
        self.collection = get_collection(self.client)
        self.caption_collection = get_caption_collection(self.client)
        self.embed_model = get_embedding_model()
        self.text_embed_model = get_text_embedding_model()
        self.florence_model, self.florence_processor = load_florence_model()
        self.cache = QueryCache()
        print("System Ready.")

    def _log(self, message, callback=None):
        """Helper to log messages to callback or stdout."""
        if callback:
            callback(message)
        else:
            print(message)

    def process_query(self, query_text, log_callback=None, timeout=None, visual_threshold=0.6, caption_threshold=0.6):
        """Process a single query with caching and logging.
        
        Args:
            query_text (str): The query text.
            log_callback (callable, optional): Function to log messages.
            timeout (float, optional): Timeout in seconds.
            visual_threshold (float): Minimum similarity for visual matches (0-1).
            caption_threshold (float): Minimum similarity for caption matches (0-1).
        """
        start_time = time.time()
        self._log(f"\nProcessing query: '{query_text}'", log_callback)

        # 1. Generate Query Embedding
        self._log("Generating query embedding...", log_callback)
        query_emb = self.embed_model.encode(query_text, normalize_embeddings=True).tolist()
        query_text_emb = self.text_embed_model.encode(query_text, normalize_embeddings=True).tolist()

        # Check timeout
        if timeout and (time.time() - start_time > timeout):
            self._log("Timeout reached during embedding generation. Returning gracefully.", log_callback)
            return []

        # 2. Check Cache
        self._log("Checking cache...", log_callback)
        cached_result = self.cache.find_similar(query_emb)
        if cached_result:
            self._log(f"--> Cache Hit! Found similar past query. Using cached results.", log_callback)
            self._display_results(cached_result, log_callback)
            return cached_result

        # Check timeout
        if timeout and (time.time() - start_time > timeout):
            self._log("Timeout reached before retrieval. Returning gracefully.", log_callback)
            return []

        # 3. Retrieval
        self._log("Retrieving top images...", log_callback)
        
        # Visual Search
        # Fetch more results (10) to allow for filtering
        visual_results = self.collection.query(query_embeddings=[query_emb], n_results=10)
        
        visual_paths = []
        if visual_results["metadatas"] and visual_results["distances"]:
            for meta, dist in zip(visual_results["metadatas"][0], visual_results["distances"][0]):
                similarity = 1 - dist
                if similarity >= visual_threshold:
                    visual_paths.append(meta["path"])
                else:
                    self._log(f"Filtered out visual match (Sim: {similarity:.2f} < {visual_threshold})", log_callback)
        
        self._log(f"Visual Search found: {len(visual_paths)} images (after filtering)", log_callback)

        # Caption Search
        caption_results = self.caption_collection.query(query_embeddings=[query_text_emb], n_results=10)
        
        caption_paths = []
        if caption_results["metadatas"] and caption_results["distances"]:
            for meta, dist in zip(caption_results["metadatas"][0], caption_results["distances"][0]):
                similarity = 1 - dist
                if similarity >= caption_threshold:
                    caption_paths.append(meta["path"])
                else:
                    self._log(f"Filtered out caption match (Sim: {similarity:.2f} < {caption_threshold})", log_callback)

        self._log(f"Caption Search found: {len(caption_paths)} images (after filtering)", log_callback)

        # Merge results (deduplicate) and track source
        candidates = {}
        for p in visual_paths:
            candidates[p] = "Visual"
        
        for p in caption_paths:
            if p in candidates:
                candidates[p] = "Visual + Caption"
            else:
                candidates[p] = "Caption"
        
        cand_paths = list(candidates.keys())
        self._log(f"Total unique candidates: {len(cand_paths)}", log_callback)

        # 4. Grounding
        self._log("Analyzing candidate images...", log_callback)
        found_any = False
        query_results = []

        for idx, img_path in enumerate(cand_paths):
            # Check timeout before processing each image
            if timeout and (time.time() - start_time > timeout):
                self._log(f"Timeout reached ({timeout}s). Stopping further processing.", log_callback)
                break

            if self.florence_model is None:
                self._log("Grounding model not available. Skipping.", log_callback)
                continue

            self._log(f"Checking {img_path}...", log_callback)
            result = run_grounding(
                self.florence_model, self.florence_processor, img_path, query_text
            )

            # Check if boxes exist
            if result and "<CAPTION_TO_PHRASE_GROUNDING>" in result:
                data = result["<CAPTION_TO_PHRASE_GROUNDING>"]
                if data.get("bboxes"):
                    output_filename = f"result_{idx}.jpg"
                    saved_path = draw_boxes(img_path, result, output_filename)
                    if saved_path:
                        self._log(f"--> Match Found! Saved visualization to {saved_path}", log_callback)
                        found_any = True
                        # Store path and source
                        query_results.append({
                            "path": saved_path,
                            "source": candidates[img_path]
                        })

        if not found_any:
            self._log("No matching object found in the top retrieved images.", log_callback)
        
        # Add to cache (even if empty, to avoid re-processing same failed query)
        self.cache.add(query_emb, query_results)
        return query_results

    def _display_results(self, results, log_callback=None):
        if not results:
            self._log("Cached result was empty (no matches found previously).", log_callback)
        else:
            for item in results:
                self._log(f"--> Cached Match: {item['path']} ({item['source']})", log_callback)


def main():
    parser = argparse.ArgumentParser(
        description="Query images for objects using RAG with Caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", type=str, help="Text query to search for objects (e.g., 'a red car')"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode instead of single query",
    )

    args = parser.parse_args()

    # Initialize System
    rag_system = RAGQuerySystem()

    if args.query:
        rag_system.process_query(args.query)
    elif args.interactive:
        print("\n--- Interactive Mode ---")
        print("Enter a query (e.g., 'a red car') or 'q' to quit.")
        while True:
            query_text = input("\nQuery: ").strip()
            if query_text.lower() == "q":
                break
            if not query_text:
                continue
            rag_system.process_query(query_text)
    else:
        # Default to interactive
        print("\n--- Interactive Mode ---")
        print("Enter a query (e.g., 'a red car') or 'q' to quit.")
        while True:
            query_text = input("\nQuery: ").strip()
            if query_text.lower() == "q":
                break
            if not query_text:
                continue
            rag_system.process_query(query_text)


if __name__ == "__main__":
    main()
