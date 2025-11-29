import pytest
from unittest.mock import MagicMock, patch
import sys
import os
import warnings

# Suppress warnings for clean test output
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*You are using a Python version.*which Google will stop supporting.*"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*resume_download.*is deprecated.*"
)

# Add the project root to the python path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture
def mock_chroma_client():
    with patch("rag_over_images.utils.chromadb.PersistentClient") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        yield client_instance

@pytest.fixture
def mock_collection():
    collection = MagicMock()
    return collection

@pytest.fixture
def mock_genai_model():
    with patch("google.generativeai.GenerativeModel") as mock:
        model_instance = MagicMock()
        mock.return_value = model_instance
        yield model_instance
