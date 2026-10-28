// Client for the Flask API in backend/app.py.

const DEFAULT_API_BASE_URL = "http://127.0.0.1:5000";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(
  /\/+$/,
  "",
);

export class ApiError extends Error {
  constructor(message, { isNetworkError = false } = {}) {
    super(message);
    this.name = "ApiError";
    this.isNetworkError = isNetworkError;
  }
}

async function send(path, options) {
  try {
    return await fetch(`${API_BASE_URL}${path}`, options);
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError(
      `Can't reach the prediction server at ${API_BASE_URL}. Make sure the Flask backend is running.`,
      { isNetworkError: true },
    );
  }
}

/** Resolve to true if GET /health answers {"status": "ok"}, otherwise false. */
export async function checkHealth(signal) {
  try {
    const response = await send("/health", { signal });
    if (!response.ok) return false;
    const body = await response.json();
    return body?.status === "ok";
  } catch {
    return false;
  }
}

function isValidPrediction(body) {
  return (
    body !== null &&
    typeof body === "object" &&
    Number.isInteger(body.predicted_digit) &&
    body.predicted_digit >= 0 &&
    body.predicted_digit <= 9 &&
    Array.isArray(body.probabilities) &&
    body.probabilities.length === 10 &&
    body.probabilities.every((p) => typeof p === "number")
  );
}

/**
 * Upload a 28x28 PNG to POST /predict.
 * Resolves to {predicted_digit, probabilities} or throws an ApiError.
 */
export async function requestPrediction(pngBlob, signal) {
  const formData = new FormData();
  formData.append("image", pngBlob, "digit.png");

  // No Content-Type header: the browser sets multipart/form-data with the boundary itself.
  const response = await send("/predict", { method: "POST", body: formData, signal });
  const body = await response.json().catch(() => null);

  if (!response.ok) {
    throw new ApiError(body?.error ?? `The server returned an error (HTTP ${response.status}).`);
  }
  if (!isValidPrediction(body)) {
    throw new ApiError("The server sent a response in an unexpected format.");
  }
  return body;
}
