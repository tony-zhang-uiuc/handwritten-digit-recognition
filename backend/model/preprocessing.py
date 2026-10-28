"""Image preprocessing shared by training and inference.

Training and prediction must transform images in exactly the same way;
otherwise the model sees inputs at prediction time that look different
from what it learned on. Both train.py and predict.py import from here.
"""

from torchvision import transforms

# Mean and standard deviation of all MNIST training pixels (after scaling to 0-1).
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

IMAGE_SIZE = 28


def get_transform():
    """Convert a 28x28 grayscale PIL image into a normalized (1, 28, 28) tensor."""
    return transforms.Compose([
        transforms.ToTensor(),  # pixels 0-255 -> 0.0-1.0
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
