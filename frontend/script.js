const API_URL = "http://127.0.0.1:8000/predict";
const MAX_HISTORY = 5;

const history = [];

async function analyze() {
  const input = document.getElementById("review-input");
  const text = input.value.trim();

  hideAll();

  if (!text) {
    showError("Please enter some text before analysing.");
    return;
  }

  const btn = document.getElementById("analyze-btn");
  btn.disabled = true;
  btn.textContent = "Analysing...";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Server error");
    }

    const data = await response.json();
    showResult(data);
    addToHistory(text, data);
  } catch (err) {
    if (err.message === "Failed to fetch") {
      showError("Cannot reach the API. Make sure the server is running: uvicorn main:app --reload");
    } else {
      showError(err.message);
    }
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyse";
  }
}

function showResult(data) {
  const card = document.getElementById("result-card");
  const label = document.getElementById("sentiment-label");
  const badge = document.getElementById("confidence-badge");
  const cleaned = document.getElementById("cleaned-text");

  const isPositive = data.sentiment === "Positive";
  label.textContent = isPositive ? "Positive" : "Negative";
  badge.textContent = `${data.confidence}% confidence`;
  cleaned.textContent = data.cleaned_text.slice(0, 120) + (data.cleaned_text.length > 120 ? "..." : "");

  card.className = `card result-card ${isPositive ? "positive" : "negative"}`;
  card.hidden = false;
}

function addToHistory(text, data) {
  history.unshift({ text, ...data });
  if (history.length > MAX_HISTORY) history.pop();
  renderHistory();
}

function renderHistory() {
  const card = document.getElementById("history-card");
  const tbody = document.getElementById("history-body");

  tbody.innerHTML = history
    .map(
      (item) => `
      <tr>
        <td>${item.text.slice(0, 60)}${item.text.length > 60 ? "…" : ""}</td>
        <td class="tag-${item.sentiment.toLowerCase()}">${item.sentiment}</td>
        <td>${item.confidence}%</td>
      </tr>`
    )
    .join("");

  card.hidden = false;
}

function showError(message) {
  const card = document.getElementById("error-card");
  document.getElementById("error-message").textContent = message;
  card.hidden = false;
}

function hideAll() {
  document.getElementById("result-card").hidden = true;
  document.getElementById("error-card").hidden = true;
}

function clearAll() {
  document.getElementById("review-input").value = "";
  hideAll();
}

document.getElementById("review-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.ctrlKey) analyze();
});
