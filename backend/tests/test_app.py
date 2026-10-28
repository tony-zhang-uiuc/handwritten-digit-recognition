"""Tests for the Flask API.

Run from the backend/ folder:
    python -m pytest

Most tests replace the real model with mocks, so they run fast and don't need
a GPU or a trained checkpoint. One test at the bottom uses the real checkpoint
and is skipped if it hasn't been trained yet.
"""

import io
from unittest.mock import patch

import pytest
from PIL import Image

import app as app_module
from model.predict import DEFAULT_CHECKPOINT

FAKE_MODEL = object()
FAKE_PROBABILITIES = [0.0] * 10
FAKE_PROBABILITIES[7] = 1.0


def make_image_file(size=(28, 28), image_format="PNG", mode="L"):
    """Create an in-memory image file, like one a browser would upload."""
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format=image_format)
    buffer.seek(0)
    return buffer


def upload(client, file, filename="digit.png"):
    return client.post(
        "/predict",
        data={"image": (file, filename)},
        content_type="multipart/form-data",
    )


@pytest.fixture
def mocks():
    """Replace model loading and inference with fakes for the duration of a test."""
    with patch.object(app_module, "load_model", return_value=FAKE_MODEL) as mock_load, \
         patch.object(app_module, "predict_digit", return_value=(7, FAKE_PROBABILITIES)) as mock_predict:
        yield mock_load, mock_predict


@pytest.fixture
def client(mocks):
    flask_app = app_module.create_app()
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


# ----- /health -----

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


# ----- /predict: successful requests -----

def test_predict_returns_digit_and_probabilities(client, mocks):
    _, mock_predict = mocks

    response = upload(client, make_image_file())

    assert response.status_code == 200
    body = response.get_json()
    assert body["predicted_digit"] == 7
    assert body["probabilities"] == FAKE_PROBABILITIES
    assert len(body["probabilities"]) == 10

    image_arg = mock_predict.call_args.args[0]
    assert image_arg.size == (28, 28)
    assert mock_predict.call_args.kwargs["model"] is FAKE_MODEL


def test_predict_accepts_jpeg(client):
    response = upload(client, make_image_file(image_format="JPEG"), filename="digit.jpg")
    assert response.status_code == 200


def test_predict_accepts_color_image(client):
    response = upload(client, make_image_file(mode="RGBA"))
    assert response.status_code == 200


def test_model_is_loaded_only_once(client, mocks):
    mock_load, mock_predict = mocks

    upload(client, make_image_file())
    upload(client, make_image_file())

    assert mock_load.call_count == 1
    assert mock_predict.call_count == 2


# ----- /predict: invalid requests -----

def test_missing_file_returns_400(client):
    response = client.post("/predict", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_empty_filename_returns_400(client):
    response = upload(client, make_image_file(), filename="")
    assert response.status_code == 400


def test_wrong_extension_returns_400(client):
    response = upload(client, make_image_file(), filename="digit.txt")
    assert response.status_code == 400
    assert "Unsupported file type" in response.get_json()["error"]


def test_non_image_content_returns_400(client):
    response = upload(client, io.BytesIO(b"this is not an image"), filename="digit.png")
    assert response.status_code == 400
    assert "not a valid image" in response.get_json()["error"]


def test_disallowed_image_format_returns_400(client):
    response = upload(client, make_image_file(image_format="GIF"), filename="digit.png")
    assert response.status_code == 400


@pytest.mark.parametrize("size", [(27, 28), (28, 27), (32, 32), (280, 280)])
def test_wrong_dimensions_return_400(client, mocks, size):
    _, mock_predict = mocks

    response = upload(client, make_image_file(size=size))

    assert response.status_code == 400
    assert "28x28" in response.get_json()["error"]
    mock_predict.assert_not_called()


def test_wrong_method_returns_json_405(client):
    response = client.get("/predict")
    assert response.status_code == 405
    assert "error" in response.get_json()


# ----- CORS -----

def test_cors_allows_vite_dev_server(client):
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"


def test_cors_rejects_other_origins(client):
    response = client.get("/health", headers={"Origin": "http://evil.example.com"})
    assert "Access-Control-Allow-Origin" not in response.headers


# ----- Real model (no mocks) -----

@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="model has not been trained yet")
def test_predict_with_real_model():
    flask_app = app_module.create_app()
    client = flask_app.test_client()

    response = upload(client, make_image_file())

    assert response.status_code == 200
    body = response.get_json()
    assert body["predicted_digit"] in range(10)
    assert len(body["probabilities"]) == 10
    assert sum(body["probabilities"]) == pytest.approx(1.0, abs=1e-4)
