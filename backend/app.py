"""Flask REST API that serves predictions from the trained digit CNN.

Run from the backend/ folder (with the virtual environment activated):
    python app.py

Endpoints:
    GET  /health   -> {"status": "ok"}
    POST /predict  -> multipart/form-data with a 28x28 PNG/JPEG in the field "image"
"""

from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from model.predict import load_model, predict_digit
from model.preprocessing import IMAGE_SIZE

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_FORMATS = {"PNG", "JPEG"}
MAX_UPLOAD_BYTES = 1 * 1024 * 1024

# Vite's development server runs on port 5173 by default.
FRONTEND_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def error_response(message, status):
    return jsonify({"error": message}), status


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    CORS(app, origins=FRONTEND_ORIGINS)

    # Load the weights once at startup; every request reuses this model.
    model = load_model()

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/predict")
    def predict():
        file = request.files.get("image")
        if file is None:
            return error_response(
                "No image uploaded. Send it as multipart/form-data in a field named 'image'.", 400
            )
        if not file.filename:
            return error_response("The uploaded file has no filename.", 400)

        extension = Path(file.filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            return error_response(
                f"Unsupported file type '{extension or 'none'}'. Upload a PNG or JPEG image.", 400
            )

        try:
            image = Image.open(file.stream)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
            return error_response("The uploaded file is not a valid image.", 400)

        # The extension can lie, so also check the format Pillow actually detected.
        if image.format not in ALLOWED_FORMATS:
            return error_response("The uploaded file is not a PNG or JPEG image.", 400)

        width, height = image.size
        if (width, height) != (IMAGE_SIZE, IMAGE_SIZE):
            return error_response(
                f"Image must be {IMAGE_SIZE}x{IMAGE_SIZE} pixels, got {width}x{height}.", 400
            )

        try:
            image.load()
        except OSError:
            return error_response("The uploaded image is corrupted or incomplete.", 400)

        digit, probabilities = predict_digit(image, model=model)
        return jsonify({"predicted_digit": digit, "probabilities": probabilities})

    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        # Return JSON instead of Flask's default HTML pages (e.g. 404, 405, 413).
        return error_response(error.description, error.code)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000)
