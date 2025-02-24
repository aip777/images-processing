import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from main import app, UPLOAD_FOLDER
from db.models import ImageMetadata
from iprocessor.processor import ImageProcessor
import os
import shutil

@pytest.fixture
def client():
    client = TestClient(app)
    yield client
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)

@pytest.fixture
def mock_db():
    db = MagicMock()
    yield db
    db.close()

@pytest.fixture
def mock_image_processor():
    processor = MagicMock(spec=ImageProcessor)
    yield processor

def test_upload_image(client, mock_db, mock_image_processor):
    file_data = {
        'file': ('test_image.tiff', b'test file data')
    }
    mock_image_processor.extract_metadata.return_value = {"key": "value"}
    mock_db.add.return_value = None
    mock_db.commit.return_value = None
    response = client.post("/upload", files=file_data, params={}, follow_redirects=True)
    assert response.status_code == 200
    assert response.json() == {"message": "Image uploaded", "metadata": {"key": "value"}}
    mock_image_processor.extract_metadata.assert_called_once()

def test_get_metadata(client, mock_db):
    mock_db.query.return_value.all.return_value = [ImageMetadata(file_name="test_image.tiff", file_path="test_path", image_metadata="{'key': 'value'}")]
    response = client.get("/metadata")
    assert response.status_code == 200
    assert "metadata" in response.json()

def test_get_slice(client, mock_db, mock_image_processor):
    mock_db.query.return_value.order_by.return_value.first.return_value = ImageMetadata(file_name="test_image.tiff", file_path="test_path", image_metadata="{'key': 'value'}")
    mock_image_processor.get_slice.return_value = [1, 2, 3]
    response = client.get("/slice?z=1&time=1&channel=1")
    assert response.status_code == 200
    assert response.json() == {"slice": [1, 2, 3]}

def test_analyze_image(client, mock_db, mock_image_processor):
    mock_db.query.return_value.order_by.return_value.first.return_value = ImageMetadata(file_name="test_image.tiff", file_path="test_path", image_metadata="{'key': 'value'}")
    mock_image_processor.perform_pca.return_value = [0.1, 0.2, 0.3]
    response = client.post("/analyze")
    assert response.status_code == 200
    assert response.json() == {"pca_result": [0.1, 0.2, 0.3]}

def test_get_statistics(client, mock_db, mock_image_processor):
    mock_db.query.return_value.order_by.return_value.first.return_value = ImageMetadata(file_name="test_image.tiff", file_path="test_path", image_metadata="{'key': 'value'}")
    mock_image_processor.compute_statistics.return_value = {"mean": 0.5, "std": 0.1}
    response = client.get("/statistics")
    assert response.status_code == 200
    assert response.json() == {"statistics": {"mean": 0.5, "std": 0.1}}
