import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import os


# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_chroma_client, get_collection, get_embedding_model

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
    print(f"Loading Florence-2 model ({FLORENCE_MODEL_ID})...")
    # trust_remote_code=True is required for Florence-2
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL_ID,
            trust_remote_code=True,
        ).to(DEVICE)
    processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True)
    return model, processor


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
    # Typical format: <CAPTION_TO_PHRASE_GROUNDING> {text_query}
    # Or just use the task token if fine-tuned/supported.
    # Florence-2 supports <CAPTION_TO_PHRASE_GROUNDING> or <OPEN_VOCABULARY_DETECTION>
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    prompt = task_prompt + text_query

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        DEVICE, torch.float32
    )  # float32 for cpu compatibility if needed, or stick to default

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


def main_query_loop():
    print("Initializing Query System...")

    # Load Retrieval Resources
    client = get_chroma_client()
    collection = get_collection(client)
    embed_model = get_embedding_model()

    # Load Grounding Resources
    florence_model, florence_processor = load_florence_model()

    print("\n--- System Ready ---")
    print("Enter a query (e.g., 'a red car') or 'q' to quit.")

    while True:
        query_text = input("\nQuery: ").strip()
        if query_text.lower() == "q":
            break
        if not query_text:
            continue

        print(f"Retrieving top images for: '{query_text}'...")

        # 1. Retrieval
        query_emb = embed_model.encode(query_text).tolist()
        results = collection.query(query_embeddings=[query_emb], n_results=3)  # Top-3

        # results['metadatas'] is a list of lists (one per query)
        cand_paths = [m["path"] for m in results["metadatas"][0]]

        # 2. Grounding
        print("Analyzing candidate images...")
        found_any = False

        for idx, img_path in enumerate(cand_paths):
            print(f"Checking {img_path}...")
            result = run_grounding(
                florence_model, florence_processor, img_path, query_text
            )

            # Check if boxes exist
            if result and "<CAPTION_TO_PHRASE_GROUNDING>" in result:
                data = result["<CAPTION_TO_PHRASE_GROUNDING>"]
                if data.get("bboxes"):
                    output_filename = f"reuslt_{idx}.jpg"
                    saved_path = draw_boxes(img_path, result, output_filename)
                    if saved_path:
                        print(f"--> Match Found! Saved visualization to {saved_path}")
                        found_any = True

        if not found_any:
            print("No matching object found in the top retrieved images.")


if __name__ == "__main__":
    main_query_loop()
