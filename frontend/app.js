const API_BASE = "http://127.0.0.1:8000";
const form = document.getElementById("prediction-form");
const resultContent = document.getElementById("result-content");

function asNumber(id) {
  return Number(document.getElementById(id).value);
}

function renderResult(data) {
  const probabilityText =
    typeof data.probability === "number"
      ? `${(data.probability * 100).toFixed(2)}%`
      : "N/A";

  const riskClass = data.prediction === 1 ? "risk" : "ok";
  const riskLabel = data.prediction === 1 ? "Likely Disease" : "Likely No Disease";

  resultContent.innerHTML = `
    <p>
      <span class="badge ${riskClass}">${riskLabel}</span>
      <span class="badge">Prediction: ${data.prediction}</span>
    </p>
    <p><strong>Estimated Probability:</strong> ${probabilityText}</p>
  `;
}

function renderError(message) {
  resultContent.innerHTML = `<p class="badge risk">Error</p><p>${message}</p>`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    age: asNumber("age"),
    sex: asNumber("sex"),
    cp: asNumber("cp"),
    trestbps: asNumber("trestbps"),
    chol: asNumber("chol"),
    fbs: asNumber("fbs"),
    restecg: asNumber("restecg"),
    thalach: asNumber("thalach"),
    exang: asNumber("exang"),
    oldpeak: asNumber("oldpeak"),
    slope: asNumber("slope"),
    ca: asNumber("ca"),
    thal: asNumber("thal"),
  };

  resultContent.innerHTML = "<p class=\"muted\">Running prediction...</p>";

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let detail = "Request failed.";
      try {
        const errData = await response.json();
        detail = errData.detail || detail;
      } catch (_) {
        // Keep fallback message when body is not JSON.
      }
      throw new Error(detail);
    }

    const data = await response.json();
    renderResult(data);
  } catch (error) {
    renderError(error.message || "Unexpected error while calling API.");
  }
});
