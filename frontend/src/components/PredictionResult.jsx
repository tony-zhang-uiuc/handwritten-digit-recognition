function formatPercent(probability) {
  const percent = probability * 100;
  if (percent > 0 && percent < 0.1) return "<0.1%";
  return `${percent.toFixed(1)}%`;
}

/** Shows the predicted digit, its confidence, and a bar for each digit 0-9. */
export default function PredictionResult({ result, isLoading, previewUrl }) {
  if (!result) {
    return (
      <div className="result-empty">
        {isLoading ? (
          <p>
            <span className="spinner" aria-hidden="true" /> Asking the model…
          </p>
        ) : (
          <p>Draw a digit and press Predict to see what the model thinks it is.</p>
        )}
      </div>
    );
  }

  const { predicted_digit: digit, probabilities } = result;

  return (
    <div className={isLoading ? "result is-updating" : "result"}>
      <div className="result-summary">
        <div className="predicted-digit" aria-label={`Predicted digit: ${digit}`}>
          {digit}
        </div>
        <div className="confidence">
          <span className="confidence-label">Confidence</span>
          <span className="confidence-value">{formatPercent(probabilities[digit])}</span>
        </div>
        {previewUrl && (
          <figure className="model-input">
            <img
              src={previewUrl}
              width="56"
              height="56"
              alt="The 28 by 28 pixel image that was sent to the model"
            />
            <figcaption>Model input (28×28)</figcaption>
          </figure>
        )}
      </div>

      <h3 className="probabilities-heading">All probabilities</h3>
      <ul className="probabilities">
        {probabilities.map((probability, d) => (
          <li
            key={d}
            className={d === digit ? "is-top" : undefined}
            aria-label={`Digit ${d}: ${formatPercent(probability)}`}
          >
            <span className="probability-digit" aria-hidden="true">
              {d}
            </span>
            <span className="probability-bar" aria-hidden="true">
              <span style={{ width: `${probability * 100}%` }} />
            </span>
            <span className="probability-value" aria-hidden="true">
              {formatPercent(probability)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
