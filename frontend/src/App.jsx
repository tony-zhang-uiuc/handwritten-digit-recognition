import { useEffect, useRef, useState } from "react";
import DrawingCanvas from "./components/DrawingCanvas.jsx";
import PredictionResult from "./components/PredictionResult.jsx";
import { API_BASE_URL, checkHealth, requestPrediction } from "./lib/api.js";
import { canvasToMnistPng } from "./lib/mnistPreprocess.js";

const API_STATUS_TEXT = {
  checking: "Checking API…",
  online: "API online",
  offline: "API offline",
};

export default function App() {
  const canvasRef = useRef(null);
  const requestRef = useRef(null); // AbortController for the prediction in flight

  const [hasDrawing, setHasDrawing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");
  const [apiStatus, setApiStatus] = useState("checking");

  useEffect(() => {
    const controller = new AbortController();
    checkHealth(controller.signal).then((ok) => {
      if (!controller.signal.aborted) setApiStatus(ok ? "online" : "offline");
    });
    return () => controller.abort();
  }, []);

  // Free the previous preview image's memory whenever it's replaced.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function handleDraw() {
    setHasDrawing(true);
    setError("");
  }

  function handleClear() {
    requestRef.current?.abort();
    requestRef.current = null;
    canvasRef.current.clear();
    setHasDrawing(false);
    setIsLoading(false);
    setResult(null);
    setError("");
    setPreviewUrl("");
  }

  async function handlePredict() {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setIsLoading(true);
    setError("");

    try {
      const png = await canvasToMnistPng(canvasRef.current.getCanvas());
      if (!png) {
        setResult(null);
        setError("The canvas is empty. Draw a digit first.");
        return;
      }
      setPreviewUrl(URL.createObjectURL(png));

      const prediction = await requestPrediction(png, controller.signal);
      setResult(prediction);
      setApiStatus("online");
    } catch (err) {
      if (err.name === "AbortError") return;
      setResult(null);
      setError(err.message || "Something went wrong while predicting.");
      if (err.isNetworkError) setApiStatus("offline");
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsLoading(false);
      }
    }
  }

  const canClear = hasDrawing || result !== null || error !== "";

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Handwritten Digit Recognition</h1>
          <p className="subtitle">
            Draw a digit from 0 to 9 and a convolutional neural network trained on MNIST will
            recognize it.
          </p>
        </div>
        <span className={`api-status api-status--${apiStatus}`} role="status">
          <span className="api-status-dot" aria-hidden="true" />
          {API_STATUS_TEXT[apiStatus]}
        </span>
      </header>

      <main className="layout">
        <section className="card" aria-labelledby="draw-heading">
          <h2 id="draw-heading">Draw</h2>
          <DrawingCanvas ref={canvasRef} onDraw={handleDraw} />
          <div className="actions">
            <button
              type="button"
              className="button button--secondary"
              onClick={handleClear}
              disabled={!canClear}
              aria-label="Clear the drawing and the prediction"
            >
              Clear
            </button>
            <button
              type="button"
              className="button button--primary"
              onClick={handlePredict}
              disabled={!hasDrawing || isLoading}
              aria-label={isLoading ? "Predicting the drawn digit" : "Predict the drawn digit"}
              aria-busy={isLoading}
            >
              {isLoading && <span className="spinner" aria-hidden="true" />}
              {isLoading ? "Predicting…" : "Predict"}
            </button>
          </div>
          <p className="hint">
            Use your mouse, finger, or a stylus. Draw one large digit near the middle.
          </p>
        </section>

        <section className="card" aria-labelledby="result-heading">
          <h2 id="result-heading">Prediction</h2>
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          <div aria-live="polite">
            <PredictionResult result={result} isLoading={isLoading} previewUrl={previewUrl} />
          </div>
        </section>
      </main>

      <footer className="app-footer">
        PyTorch CNN · Flask API at <code>{API_BASE_URL}</code>
      </footer>
    </div>
  );
}
