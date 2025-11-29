import os
import google.generativeai as genai
import json
import random

def generate_query_suggestions(captions, api_key):
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
        
        # --- FIX APPLIED HERE: Changed model name to a currently recognized one ---
        # Using gemini-2.5-flash as the currently available, best Flash model
        model = genai.GenerativeModel('gemini-2.5-flash') 

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