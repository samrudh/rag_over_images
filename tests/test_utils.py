import pytest
from unittest.mock import MagicMock, patch
from rag_over_images.utils import get_chroma_client, get_collection, manage_collection_limit

def test_get_chroma_client(mock_chroma_client):
    client = get_chroma_client()
    assert client == mock_chroma_client

def test_get_collection(mock_chroma_client, mock_collection):
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    # get_collection only takes client, name is hardcoded or internal?
    # Checking utils.py: def get_collection(client: ClientAPI) -> Collection:
    # It hardcodes "image_embeddings".
    collection = get_collection(mock_chroma_client)
    assert collection == mock_collection
    # The previous failure said "expected call not found". 
    # Maybe get_or_create_collection is called with different args or not called on the mock we expect.
    # Let's check utils.py again.
    # client.get_or_create_collection(name="image_embeddings")
    # If this fails, it might be because of metadata arg.
    # utils.py: name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    mock_chroma_client.get_or_create_collection.assert_called_with(
        name="image_embeddings", metadata={"hnsw:space": "cosine"}
    )

def test_manage_collection_limit(mock_chroma_client, mock_collection):
    # Setup mock data
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    mock_collection.count.return_value = 150
    mock_collection.get.return_value = {
        "ids": ["1", "2", "3"],
        "metadatas": [{"timestamp": 100}, {"timestamp": 200}, {"timestamp": 300}]
    }
    
    # Signature: manage_collection_limit(client: ClientAPI, limit: int, new_count: int)
    manage_collection_limit(mock_chroma_client, limit=100, new_count=0)
    
    # 150 existing + 0 new > 100 limit. Delete 50.
    
    mock_collection.delete.assert_called()
    call_args = mock_collection.delete.call_args
    assert "ids" in call_args[1]
    assert len(call_args[1]["ids"]) == 3
    assert call_args[1]["ids"] == ["1", "2", "3"] 

def test_manage_collection_limit_no_delete(mock_chroma_client, mock_collection):
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    mock_collection.count.return_value = 50
    manage_collection_limit(mock_chroma_client, limit=100, new_count=0)
    mock_collection.delete.assert_not_called()
