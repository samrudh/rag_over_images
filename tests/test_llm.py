import pytest
from unittest.mock import MagicMock, patch
from rag_over_images.LLM import generate_query_suggestions, validate_search_results

def test_generate_query_suggestions(mock_genai_model):
    # Setup mock response
    mock_response = MagicMock()
    mock_response.text = '["query 1", "query 2"]'
    mock_genai_model.generate_content.return_value = mock_response
    
    captions = ["caption 1", "caption 2"]
    api_key = "test_key"
    
    suggestions = generate_query_suggestions(captions, api_key)
    
    assert suggestions == ["query 1", "query 2"]
    mock_genai_model.generate_content.assert_called_once()

def test_generate_query_suggestions_no_key():
    suggestions = generate_query_suggestions(["caption"], "")
    assert suggestions == []

def test_validate_search_results(mock_genai_model):
    # Setup mock response
    mock_response = MagicMock()
    mock_response.text = "Validation successful"
    mock_response.parts = [MagicMock()] # Simulate valid parts
    mock_response.candidates = [MagicMock()] # Simulate valid candidates
    mock_genai_model.generate_content.return_value = mock_response
    
    # Patch PIL.Image.open where it is used. Since it is imported inside the function,
    # we might need to patch 'PIL.Image.open' globally or mock the module.
    # Actually, if it's imported as 'from PIL import Image', we should patch 'PIL.Image.open'.
    with patch("PIL.Image.open") as mock_open:
        mock_open.return_value = MagicMock()
        
        result = validate_search_results("query", ["path/to/image.jpg"], "test_key")
        
        assert result == "Validation successful"
        mock_genai_model.generate_content.assert_called_once()

def test_validate_search_results_blocked(mock_genai_model):
    # Setup mock response for blocked content
    mock_response = MagicMock()
    mock_response.parts = []
    mock_response.prompt_feedback = "Blocked due to safety"
    mock_genai_model.generate_content.return_value = mock_response
    
    with patch("PIL.Image.open") as mock_open:
        mock_open.return_value = MagicMock()
        
        result = validate_search_results("query", ["path/to/image.jpg"], "test_key")
        
        assert "Validation blocked" in result
