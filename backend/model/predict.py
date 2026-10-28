"""Load the trained DigitCNN and predict the digit in a 28x28 grayscale image.

Use from other Python code (e.g. the future Flask API):
    from model.predict import predict_digit
    digit, probabilities = predict_digit(image)

Or try it from the command line (run from the backend/ folder):
    python -m model.predict --index 0
    python -m model.predict --image path/to/digit.png
"""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

from model.cnn import DigitCNN
from model.preprocessing import IMAGE_SIZE, get_transform

MODEL_DIR = Path(__file__).resolve().parent
DATA_DIR = MODEL_DIR / "data"
DEFAULT_CHECKPOINT = MODEL_DIR / "checkpoints" / "mnist_cnn.pth"


def load_model(checkpoint_path=DEFAULT_CHECKPOINT, device=None):
    """Build a DigitCNN, load the trained weights, and switch to evaluation mode."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No trained weights found at {checkpoint_path}. "
            "Train the model first with: python -m model.train"
        )

    model = DigitCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


@lru_cache(maxsize=1)
def _default_model():
    # Load the weights once and reuse them instead of reading the file on every prediction.
    return load_model()


def _to_grayscale_image(image):
    """Convert a PIL image or NumPy array into a 28x28 8-bit grayscale PIL image."""
    if isinstance(image, Image.Image):
        image = image.convert("L")
    else:
        array = np.asarray(image)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D grayscale array, got shape {array.shape}")
        if np.issubdtype(array.dtype, np.floating):
            # Float pixels are expected in the range 0.0-1.0.
            array = np.clip(array, 0.0, 1.0) * 255
        array = np.clip(array, 0, 255).round().astype(np.uint8)
        image = Image.fromarray(array)

    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        width, height = image.size
        raise ValueError(f"Expected a {IMAGE_SIZE}x{IMAGE_SIZE} image, got {width}x{height}")
    return image


def predict_digit(image, model=None):
    """Predict which digit (0-9) is in a 28x28 grayscale image.

    Args:
        image: a PIL image, or a 2D NumPy array with integer values 0-255
            or float values 0.0-1.0. Like MNIST, the digit should be
            white (bright) on a black (dark) background.
        model: an already-loaded model. If omitted, the default checkpoint is used.

    Returns:
        (digit, probabilities): the predicted digit as an int, and a list of
        10 floats where probabilities[i] is the model's confidence that the digit is i.
    """
    if model is None:
        model = _default_model()
    device = next(model.parameters()).device

    tensor = get_transform()(_to_grayscale_image(image))  # shape (1, 28, 28)
    batch = tensor.unsqueeze(0).to(device)  # the model expects a batch: (1, 1, 28, 28)

    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)[0]

    digit = int(probabilities.argmax())
    return digit, probabilities.tolist()


def main():
    parser = argparse.ArgumentParser(description="Predict a digit with the trained CNN.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--index", type=int, default=0, help="index of an MNIST test image (default: 0)")
    source.add_argument("--image", type=str, help="path to a 28x28 grayscale image file")
    args = parser.parse_args()

    if args.image:
        image = Image.open(args.image)
        true_label = None
    else:
        test_set = datasets.MNIST(DATA_DIR, train=False, download=True)
        image, true_label = test_set[args.index]

    digit, probabilities = predict_digit(image)

    print(f"Predicted digit: {digit}")
    if true_label is not None:
        print(f"True label:      {true_label}")
    print("Probabilities:")
    for d, p in enumerate(probabilities):
        print(f"  {d}: {p:.4f}")


if __name__ == "__main__":
    main()
