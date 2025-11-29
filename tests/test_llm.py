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
    # Return a valid JSON string
    mock_response.text = '{"explanation": "Validation successful", "valid_indices": [0]}'
    mock_response.parts = [MagicMock()] 
    mock_response.candidates = [MagicMock()]
    mock_genai_model.generate_content.return_value = mock_response
    
    with patch("PIL.Image.open") as mock_open:
        mock_open.return_value = MagicMock()
        
        explanation, indices = validate_search_results("query", ["path/to/image.jpg"], "test_key")
        
        assert explanation == "Validation successful"
        assert indices == [0]
        mock_genai_model.generate_content.assert_called_once()

@patch("rag_over_images.LLM.genai")
def test_validate_search_results_blocked(mock_genai):
    # Setup
    mock_model = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    
    # Mock blocked response
    mock_response = MagicMock()
    mock_response.parts = []
    mock_response.prompt_feedback = "Blocked due to safety"
    mock_model.generate_content.return_value = mock_response
    
    with patch("PIL.Image.open") as mock_open:
        mock_open.return_value = MagicMock()

        explanation, indices = validate_search_results("query", ["path/to/image.jpg"], "test_key")

        assert "Validation blocked" in explanation
        assert len(indices) == 1 # Should return all indices if blocked/failed to be safe
