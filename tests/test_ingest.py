import pytest
from unittest.mock import MagicMock, patch
from rag_over_images.ingestion.ingest import ingest_images

@patch("rag_over_images.ingestion.ingest.get_chroma_client")
@patch("rag_over_images.ingestion.ingest.get_embedding_model")
@patch("rag_over_images.ingestion.ingest.get_caption_model")
@patch("rag_over_images.ingestion.ingest.get_text_embedding_model")
@patch("rag_over_images.ingestion.ingest.get_caption_collection")
@patch("rag_over_images.ingestion.ingest.os.listdir")
@patch("rag_over_images.ingestion.ingest.Image.open")
def test_ingest_images(
    mock_open, mock_listdir, mock_get_caption_collection, mock_get_text_model, mock_get_caption_model, mock_get_model, mock_get_client
):
    # Setup mocks
    mock_listdir.return_value = ["image1.jpg", "image2.jpg"]
    mock_open.return_value = MagicMock() # Mock image object
    
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_collection = MagicMock()
    # ingest.py calls get_collection which calls client.get_or_create_collection
    # But wait, get_collection is imported from utils.
    # In ingest.py: collection = get_collection(client)
    # And get_collection is NOT patched in test_ingest.py (it's imported from utils).
    # So it runs the real get_collection from utils.py?
    # No, we didn't patch utils.get_collection.
    # But we patched ingest.get_chroma_client.
    # So client is a mock.
    # utils.get_collection(client) calls client.get_or_create_collection.
    # So mock_client.get_or_create_collection should be called.
    mock_client.get_or_create_collection.return_value = mock_collection
    # ingest.py calls manage_collection_limit(client, limit, new_count)
    # manage_collection_limit calls collection.count()
    # We need to set return value for count() to avoid MagicMock > int error
    mock_collection.count.return_value = 50
    
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3] # Mock embedding
    mock_get_model.return_value = mock_model
    
    mock_text_model = MagicMock()
    mock_text_model.encode.return_value = MagicMock()
    mock_text_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    mock_get_text_model.return_value = mock_text_model
    
    mock_processor = MagicMock()
    mock_caption_model = MagicMock()
    mock_get_caption_model.return_value = (mock_processor, mock_caption_model)
    
    mock_caption_collection = MagicMock()
    mock_get_caption_collection.return_value = mock_caption_collection
    
    # Mock Florence generation (BLIP actually)
    mock_processor.decode.return_value = "A cat sitting on a mat"
    
    # Run ingestion
    ingest_images(input_dir="test_dir", n_images=2, mode="append")
    
    # Verify interactions
    mock_client.get_or_create_collection.assert_called()
    # ingest.py logic:
    # 1. collection = get_collection(client) -> calls get_or_create_collection
    # 2. caption_collection = get_caption_collection(client) -> calls get_or_create_collection
    # So it's called twice.
    # But we want to check if collection.add or upsert was called.
    # The code uses upsert.
    # collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
    # caption_collection.upsert(...)
    
    # Wait, the assertion that failed was: assert mock_collection.add.call_count == 2
    # But the code uses upsert, not add.
    # And it calls upsert ONCE per batch (all images at once).
    # So mock_collection.upsert.call_count should be 1.
    # And mock_caption_collection.upsert.call_count should be 1.
    
    assert mock_collection.upsert.call_count == 1
    assert mock_caption_collection.upsert.call_count == 1
    
    # Verify clean mode
    ingest_images(input_dir="test_dir", n_images=1, mode="clean")
    mock_client.delete_collection.assert_called()
