const API_BASE = "http://127.0.0.1:8000";

async function analyze() {
  const movie = document.getElementById("movie-input").value.trim();
  const review = document.getElementById("review-input").value.trim();

  hideCards();

  if (!movie) { showError("Please enter a movie title."); return; }
  if (!review) { showError("Please write a review."); return; }

  const btn = document.getElementById("analyze-btn");
  btn.disabled = true;
  btn.textContent = "Analysing...";

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ movie, review }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Server error");
    }

    const data = await response.json();
    showResult(data);
    await loadHistory();
  } catch (err) {
    if (err.message === "Failed to fetch") {
      showError("Cannot reach the API. Make sure the server is running:\n  python -m uvicorn main:app --reload");
    } else {
      showError(err.message);
    }
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyse & Save";
  }
}

function showResult(data) {
  const card = document.getElementById("result-card");
  const isPositive = data.sentiment === "Positive";

  document.getElementById("movie-label").textContent = data.movie;
  document.getElementById("sentiment-label").textContent = `— ${data.sentiment}`;
  document.getElementById("confidence-badge").textContent = `${data.confidence}% confidence`;

  card.className = `card result-card ${isPositive ? "positive" : "negative"}`;
  card.hidden = false;
}

async function loadHistory() {
  try {
    const response = await fetch(`${API_BASE}/reviews`);
    const reviews = await response.json();
    renderHistory(reviews);
  } catch {
    // silently skip — history failing shouldn't block the main flow
  }
}

function renderHistory(reviews) {
  const emptyState = document.getElementById("empty-state");
  const table = document.getElementById("history-table");
  const tbody = document.getElementById("history-body");

  if (reviews.length === 0) {
    emptyState.hidden = false;
    table.hidden = true;
    return;
  }

  emptyState.hidden = true;
  table.hidden = false;

  tbody.innerHTML = reviews.map((r) => {
    const date = r.created_at ? r.created_at.replace("T", " ") : "";
    const reviewSnippet = r.review.length > 80 ? r.review.slice(0, 80) + "…" : r.review;
    const tag = r.sentiment === "Positive" ? "tag-positive" : "tag-negative";
    return `
      <tr>
        <td><strong>${escapeHtml(r.movie)}</strong></td>
        <td class="review-cell">${escapeHtml(reviewSnippet)}</td>
        <td class="${tag}">${r.sentiment}</td>
        <td>${r.confidence}%</td>
        <td>${date}</td>
      </tr>`;
  }).join("");
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showError(message) {
  const card = document.getElementById("error-card");
  document.getElementById("error-message").textContent = message;
  card.hidden = false;
}

function hideCards() {
  document.getElementById("result-card").hidden = true;
  document.getElementById("error-card").hidden = true;
}

function clearForm() {
  document.getElementById("movie-input").value = "";
  document.getElementById("review-input").value = "";
  hideCards();
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.ctrlKey) analyze();
});

// load existing reviews on page open
loadHistory();
