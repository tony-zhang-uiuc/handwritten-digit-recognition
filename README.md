# Handwritten Digit Recognition

A full-stack web app that recognizes handwritten digits (0–9). You draw a digit in the browser, the frontend sends it to a Flask API, and a convolutional neural network (CNN) trained with PyTorch predicts which digit it is.

> **Status:** Initial project structure only. The model, API, and frontend are not implemented yet.

## Tech Stack

| Part | Technology |
| --- | --- |
| Model training | Python, PyTorch |
| Backend API | Flask |
| Frontend | React + Vite |
| Containerization (planned) | Docker |

## Project Structure

```
handwritten-digit-recognition/
├── backend/              # Python code: Flask API and model training
│   ├── model/            # CNN definition, training script, and saved weights
│   └── requirements.txt  # Python dependencies
├── frontend/             # React + Vite app (drawing canvas and results)
├── .gitignore            # Files Git should not track
└── README.md             # This file
```

## How It Will Work

1. **Train** a CNN on the MNIST dataset of handwritten digits (`backend/model/`).
2. **Serve** predictions through a Flask REST API (`backend/`).
3. **Draw** a digit in the React app, which sends the image to the API and shows the prediction (`frontend/`).

## Getting Started (later)

Setup instructions will be added once the code exists. The general flow will be:

- **Backend:** create a Python virtual environment, then install dependencies with `pip install -r backend/requirements.txt`.
- **Frontend:** scaffold the Vite React app in `frontend/`, then run `npm install`.
- **Docker:** add a `Dockerfile` for each service and a `docker-compose.yml` at the root.

## Roadmap

- [ ] Define and train the CNN
- [ ] Build the Flask prediction endpoint
- [ ] Build the React drawing interface
- [ ] Add Docker support
