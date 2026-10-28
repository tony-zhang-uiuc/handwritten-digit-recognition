"""Train the DigitCNN on MNIST and save the best weights.

The 60,000 MNIST training images are split into 54,000 for training and
6,000 for validation. Validation accuracy picks the best epoch; the official
10,000-image test set is used only once, at the end, for the final score.

Run from the backend/ folder:
    python -m model.train
    python -m model.train --epochs 10 --lr 0.0005
"""

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from model.cnn import DigitCNN
from model.preprocessing import get_transform

MODEL_DIR = Path(__file__).resolve().parent
DATA_DIR = MODEL_DIR / "data"
CHECKPOINT_PATH = MODEL_DIR / "checkpoints" / "mnist_cnn.pth"

TRAIN_SIZE = 54_000
VAL_SIZE = 6_000


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    """Make runs repeatable by seeding every random number generator we use."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ask cuDNN for deterministic algorithms (slightly slower on GPU, but repeatable).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(batch_size, seed):
    """Return (train, validation, test) data loaders."""
    transform = get_transform()
    full_train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    # Seeded generators make the split and the shuffle order the same on every run.
    split_generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train_set, [TRAIN_SIZE, VAL_SIZE], generator=split_generator
    )

    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, generator=shuffle_generator
    )
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Run one pass over the training data and return the average loss."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    """Return the accuracy (0-100%) of the model on the given dataset."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
    return 100.0 * correct / len(loader.dataset)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_data_loaders(args.batch_size, args.seed)
    print(
        f"Training examples: {len(train_loader.dataset)} | "
        f"validation examples: {len(val_loader.dataset)} | "
        f"test examples: {len(test_loader.dataset)}"
    )

    model = DigitCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_accuracy = evaluate(model, val_loader, device)

        message = (
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"validation accuracy: {val_accuracy:.2f}%"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            message += "  <- best so far, saved"
        print(message)

    print(f"\nBest validation accuracy: {best_val_accuracy:.2f}% (epoch {best_epoch})")
    print(f"Weights saved to: {CHECKPOINT_PATH}")

    # Final, one-time evaluation of the best checkpoint on the untouched test set.
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    test_accuracy = evaluate(model, test_loader, device)
    print(f"Test accuracy of best checkpoint: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
