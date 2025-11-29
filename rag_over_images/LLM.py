import os
import google.generativeai as genai
import json
import random


from typing import List, Any

def generate_query_suggestions(captions: List[str], api_key: str) -> List[str]:
    """
    Generates a list of query suggestions based on image captions using Gemini.

    Args:
        captions (list): List of string captions.
        api_key (str): Gemini API Key.

    Returns:
        list: A list of suggested query strings.
    """
    if not api_key:
        print("No API Key provided for query suggestions.")
        return []

    try:
        # Use the standard client without manual configuration if possible,
        # or rely on the environment variable GEMINI_API_KEY.
        # If the key must be passed directly:
        genai.configure(api_key=api_key)

        # Using gemini-1.5-flash-latest as a stable alias
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        # Sample captions if there are too many to fit in context comfortably
        sample_captions = captions[:100] if len(captions) > 100 else captions

        captions_text = "\n".join([f"- {c}" for c in sample_captions])

        # Prompt optimization: Ask for a specific number and use clear format
        prompt = f"""
        I have a RAG (Retrieval Augmented Generation) system over images. 
        Here are some captions of the images in my database:
        
        {captions_text}
        
        Based on these captions, please generate a list of 50 diverse and interesting search queries that a user might ask to find these images.
        The queries should be natural language phrases.
        
        Format the output strictly as a JSON list of strings. Example: ["query 1", "query 2"]
        Do not include any other text, explanation, or markdown formatting (like ```json ... ```) in your response, just the raw JSON string.
        """

        response = model.generate_content(prompt)

        text = response.text.strip()

        # Clean up potential markdown formatting if the model ignores instruction
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            # Check for generic ``` block
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        # Strip any leading/trailing whitespace after cleanup
        text = text.strip()

        suggestions = json.loads(text)

        if isinstance(suggestions, list):
            return suggestions
        else:
            print(f"LLM did not return a list. Response text: {text[:100]}...")
            return []

    except Exception as e:
        print(f"Error generating query suggestions: {e}")
        return []


def validate_search_results(query: str, image_paths: List[str], api_key: str) -> str:
    """
    Validates if the query object is present in the provided images using Gemini.

    Args:
        query (str): The user's search query.
        image_paths (List[str]): List of paths to the candidate images.
        api_key (str): Gemini API Key.

    Returns:
        str: The validation response from the LLM.
    """
    if not api_key:
        return "Validation skipped: No API Key provided."

    try:
        genai.configure(api_key=api_key)
        # Using gemini-1.5-flash-latest as a stable alias
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        # Load images
        images = []
        from PIL import Image
        for path in image_paths:
            try:
                img = Image.open(path)
                images.append(img)
            except Exception as e:
                print(f"Error loading image for validation {path}: {e}")

        if not images:
            return "Validation skipped: No valid images to analyze."

        prompt = f"""
        As a RAG system, below are the candidate images for the query: "{query}"
        
        Please analyze them with the following criteria and convey whether the object is present or not.
        
        Criteria:
        1. Only use the images shared.
        2. NEVER use additional objects which are not present in the image to answer.
        3. Analyze carefully if what the user asked in the query is actually present in the image.
        4. If the object is NOT present, provide a two-sentence reason why.
        5. If present, briefly confirm which image(s) it appears in.
        """

        # content = [prompt, *images] # This syntax might not work directly depending on library version
        # Construct content list
        content: List[Any] = [prompt]
        content.extend(images)

        response = model.generate_content(content)
        return response.text

    except Exception as e:
        return f"Validation failed: {str(e)}"
