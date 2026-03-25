let lastQuestion = "";
let lastAnswer = "";
let lastQaId = "";
let currentSessionId = "";
let historyItems = [];
let historySessions = [];
let selectedHistorySessionId = "";
let historySessionQuery = "";
let historySessionPage = 1;
const historySessionPageSize = 10;
let historyTracePage = 1;
const historyTracePageSize = 12;
let historyTraceQuery = "";

function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function setGlobalStatus(text, isError = false) {
  const el = document.getElementById("global-status");
  el.textContent = text;
  el.classList.toggle("danger", isError);
}

function showTab(tab) {
  const askSection = document.getElementById("ask-section");
  const historySection = document.getElementById("history-section");
  const askTab = document.getElementById("tab-ask");
  const historyTab = document.getElementById("tab-history");

  if (tab === "history") {
    askSection.classList.add("hidden");
    historySection.classList.remove("hidden");
    askTab.classList.remove("active");
    historyTab.classList.add("active");
    refreshHistoryPane();
  } else {
    historySection.classList.add("hidden");
    askSection.classList.remove("hidden");
    historyTab.classList.remove("active");
    askTab.classList.add("active");
  }
}

function getStoredSessionId() {
  return localStorage.getItem("rag_session_id") || "";
}

function setStoredSessionId(sessionId) {
  currentSessionId = sessionId || "";
  localStorage.setItem("rag_session_id", currentSessionId);
}

function clearAskForm() {
  document.getElementById("q").value = "";
}

function renderSessionMessages(items) {
  const container = document.getElementById("session-messages");
  if (!items.length) {
    container.innerHTML = `<div class="muted">No messages in this session yet.</div>`;
    return;
  }
  container.innerHTML = items.map((m, i) => `
    <div class="session-item">
      <div class="session-q">${i + 1}. ${esc(m.question || "")}</div>
      <div>${esc(m.answer || "")}</div>
      <div class="session-meta">rating=${esc(m.rating ?? "-")} | trace=${esc(m.trace_id || "-")}</div>
    </div>
  `).join("");
}

async function refreshSessions() {
  try {
    const res = await fetch("/sessions");
    const data = await res.json();
    const items = data.items || [];
    const select = document.getElementById("session-select");
    const previous = currentSessionId || getStoredSessionId();

    select.innerHTML = items.map((s) => {
      const label = `${s.session_id.slice(0, 8)} (${s.message_count})`;
      return `<option value="${esc(s.session_id)}">${esc(label)}</option>`;
    }).join("");

    if (!items.length) {
      setStoredSessionId("");
      document.getElementById("session-status").textContent = "No sessions yet";
      renderSessionMessages([]);
      return;
    }

    const found = items.find((s) => s.session_id === previous);
    const selected = found ? previous : items[0].session_id;
    select.value = selected;
    setStoredSessionId(selected);
    document.getElementById("session-status").textContent = `Current: ${selected}`;
    await loadSessionMessages(selected);
  } catch (e) {
    setGlobalStatus(`Session load failed: ${e}`, true);
  }
}

async function loadSessionMessages(sessionId) {
  if (!sessionId) {
    renderSessionMessages([]);
    return;
  }
  const res = await fetch(`/sessions/${encodeURIComponent(sessionId)}/messages`);
  const data = await res.json();
  renderSessionMessages(data.items || []);
}

async function onSessionChange() {
  const select = document.getElementById("session-select");
  const id = select.value || "";
  setStoredSessionId(id);
  document.getElementById("session-status").textContent = `Current: ${id}`;
  await loadSessionMessages(id);
}

async function newChat() {
  const id = crypto.randomUUID().replaceAll("-", "");
  setStoredSessionId(id);
  document.getElementById("session-status").textContent = `Current: ${id}`;
  renderSessionMessages([]);
}

function toHistoryRow(it, idx) {
  const question = it.question || "";
  const answer = it.answer || "";
  const rating = it.rating == null ? "-" : String(it.rating);
  const evalObj = it.evaluation || {};
  const simObj = it.similarity || {};
  const generationModel = it.generation_model || "-";
  const evaluationModel = it.evaluation_model || "-";
  const askedAt = it.asked_at || "";
  const traceId = it.trace_id || "";
  const rawScores = Array.isArray(simObj.scores) ? simObj.scores : [];
  const similarityValue = simObj.best_score ?? (rawScores.length ? rawScores[0] : "-");
  const evalHtml = `
    <span class="score-chip">f: ${esc(evalObj.faithfulness ?? "-")}</span>
    <span class="score-chip">r: ${esc(evalObj.relevance ?? "-")}</span>
    <span class="score-chip">v: ${esc(evalObj.verdict ?? "-")}</span>
  `;
  const simHtml = `
    <span class="score-chip">score: ${esc(similarityValue)}</span>
    <span class="score-chip">best: ${esc(simObj.best_score ?? "-")}</span>
    <span class="score-chip">avg: ${esc(simObj.avg_score ?? "-")}</span>
    <span class="score-chip">k: ${esc(simObj.k ?? "-")}</span>
  `;
  const modelHtml = `
    <span class="score-chip">gen: ${esc(generationModel)}</span>
    <span class="score-chip">eval: ${esc(evaluationModel)}</span>
  `;

  return `
    <tr>
      <td class="history-mono">${esc(askedAt)}</td>
      <td class="history-cell-question">${esc(question)}</td>
      <td class="history-answer">${esc(answer)}</td>
      <td><strong>${esc(rating)}</strong></td>
      <td>${evalHtml}</td>
      <td>${simHtml}</td>
      <td>${modelHtml}</td>
      <td class="history-mono">${esc(traceId)}</td>
      <td>
        <div class="tiny-actions">
          <button onclick="editRow(${idx})">Edit</button>
          <button onclick="setRating(${idx},1)">1</button>
          <button onclick="setRating(${idx},2)">2</button>
          <button onclick="setRating(${idx},3)">3</button>
          <button onclick="setRating(${idx},4)">4</button>
          <button onclick="setRating(${idx},5)">5</button>
        </div>
      </td>
    </tr>
  `;
}

function getFilteredHistorySessions() {
  const q = historySessionQuery.trim().toLowerCase();
  if (!q) return historySessions;
  return historySessions.filter((s) => {
    const sessionId = String(s.session_id || "").toLowerCase();
    const title = String(s.title || "").toLowerCase();
    return sessionId.includes(q) || title.includes(q);
  });
}

function renderHistorySessions() {
  const container = document.getElementById("history-session-list");
  const pageLabel = document.getElementById("history-session-page");
  const prevBtn = document.getElementById("history-session-prev");
  const nextBtn = document.getElementById("history-session-next");
  const selectedLabel = document.getElementById("history-selected-session");

  const filtered = getFilteredHistorySessions();
  const totalPages = Math.max(1, Math.ceil(filtered.length / historySessionPageSize));
  historySessionPage = Math.min(Math.max(1, historySessionPage), totalPages);
  const start = (historySessionPage - 1) * historySessionPageSize;
  const pageItems = filtered.slice(start, start + historySessionPageSize);

  if (!pageItems.length) {
    container.innerHTML = `<div class="muted" style="padding:10px;">No sessions found.</div>`;
  } else {
    container.innerHTML = pageItems.map((s) => {
      const active = selectedHistorySessionId === s.session_id ? "active" : "";
      return `
        <button class="history-session-item ${active}" onclick="selectHistorySession('${esc(s.session_id)}')">
          <div class="history-session-id">${esc((s.session_id || "").slice(0, 16))}</div>
          <div class="history-session-title">${esc(s.title || "Untitled")} • ${esc(s.message_count ?? 0)} messages</div>
        </button>
      `;
    }).join("");
  }

  pageLabel.textContent = `Page ${historySessionPage}/${totalPages}`;
  prevBtn.disabled = historySessionPage <= 1;
  nextBtn.disabled = historySessionPage >= totalPages;
  selectedLabel.textContent = `Selected: ${selectedHistorySessionId ? selectedHistorySessionId : "all"}`;
}

function onHistorySessionSearch() {
  historySessionQuery = document.getElementById("history-session-search").value || "";
  historySessionPage = 1;
  renderHistorySessions();
}

function changeHistorySessionPage(delta) {
  historySessionPage += delta;
  renderHistorySessions();
}

function selectHistorySession(sessionId) {
  selectedHistorySessionId = sessionId || "";
  historyTracePage = 1;
  renderHistorySessions();
  renderHistoryTraces();
}

function clearHistorySessionFilter() {
  selectedHistorySessionId = "";
  historyTracePage = 1;
  renderHistorySessions();
  renderHistoryTraces();
}

function renderHistoryTraces() {
  const tbody = document.getElementById("langfuse-history-body");
  const caption = document.getElementById("history-trace-caption");
  const pageLabel = document.getElementById("history-trace-page");
  const prevBtn = document.getElementById("history-trace-prev");
  const nextBtn = document.getElementById("history-trace-next");

  const query = historyTraceQuery.trim().toLowerCase();
  const visible = historyItems
    .map((it, idx) => ({ it, idx }))
    .filter(({ it }) => !selectedHistorySessionId || it.session_id === selectedHistorySessionId)
    .filter(({ it }) => {
      if (!query) return true;
      const sim = it.similarity || {};
      const text = [
        it.trace_id || "",
        it.question || "",
        it.generation_model || "",
        it.evaluation_model || "",
        sim.best_score ?? "",
        sim.avg_score ?? "",
        Array.isArray(sim.scores) ? sim.scores.join(" ") : ""
      ]
        .join(" ")
        .toLowerCase();
      return text.includes(query);
    });

  const totalPages = Math.max(1, Math.ceil(visible.length / historyTracePageSize));
  historyTracePage = Math.min(Math.max(1, historyTracePage), totalPages);
  const start = (historyTracePage - 1) * historyTracePageSize;
  const pageItems = visible.slice(start, start + historyTracePageSize);

  if (!pageItems.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="muted">No traces for this selection.</td></tr>`;
  } else {
    tbody.innerHTML = pageItems.map(({ it, idx }) => toHistoryRow(it, idx)).join("");
  }

  pageLabel.textContent = `Page ${historyTracePage}/${totalPages}`;
  prevBtn.disabled = historyTracePage <= 1;
  nextBtn.disabled = historyTracePage >= totalPages;
  caption.textContent = selectedHistorySessionId
    ? `Showing ${visible.length} trace(s) for session ${selectedHistorySessionId}`
    : `Showing all traces (${visible.length})`;
}

function changeHistoryTracePage(delta) {
  historyTracePage += delta;
  renderHistoryTraces();
}

function onHistoryTraceSearch() {
  historyTraceQuery = document.getElementById("history-trace-search").value || "";
  historyTracePage = 1;
  renderHistoryTraces();
}

async function refreshHistorySessions() {
  const res = await fetch("/sessions");
  const data = await res.json();
  historySessions = data.items || [];
  renderHistorySessions();
}

async function refreshHistoryPane() {
  await refreshHistorySessions();
  await refreshLangfuseHistory();
}

async function editRow(idx) {
  const item = historyItems[idx] || {};
  const qaId = item.qa_id || "";
  const traceId = item.trace_id || "";
  const answer = item.answer || "";
  const rating = item.rating == null ? "" : String(item.rating);
  const newAnswer = prompt("Update answer", answer || "");
  if (newAnswer == null) return;
  const ratingIn = prompt("Update rating (1-5, blank to keep)", rating || "");
  if (ratingIn == null) return;

  const body = { qa_id: qaId || null, trace_id: traceId || null, answer: newAnswer };
  if (ratingIn.trim() !== "") body.rating = Number(ratingIn.trim());

  const res = await fetch("/history/update", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  const data = await res.json();
  const status = document.getElementById("history-status");
  if (res.ok) {
    status.textContent = "Record updated";
    await refreshLangfuseHistory();
  } else {
    status.textContent = `Update failed: ${data.detail || "unknown error"}`;
  }
}

async function setRating(idx, rating) {
  const item = historyItems[idx] || {};
  const body = { qa_id: item.qa_id || null, trace_id: item.trace_id || null, rating };
  const res = await fetch("/history/update", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  const data = await res.json();
  const status = document.getElementById("history-status");
  if (res.ok) {
    status.textContent = `Rating updated to ${rating}`;
    await refreshLangfuseHistory();
  } else {
    status.textContent = `Rating update failed: ${data.detail || "unknown error"}`;
  }
}

async function refreshLangfuseHistory() {
  const status = document.getElementById("history-status");
  status.textContent = "Loading...";
  status.classList.add("loading");
  try {
    const res = await fetch("/history/langfuse?limit=100");
    const data = await res.json();
    historyItems = data.items || [];
    historyTracePage = 1;
    renderHistoryTraces();

    if (data.error) {
      status.textContent = `Loaded ${historyItems.length} rows (warning: ${data.error})`;
    } else {
      status.textContent = `Loaded ${historyItems.length} rows`;
    }
  } catch (e) {
    status.textContent = `Failed: ${e}`;
  } finally {
    status.classList.remove("loading");
  }
}

async function ask() {
  const q = document.getElementById("q").value.trim();
  if (!q) return;

  const askStatus = document.getElementById("status");
  askStatus.textContent = "Running...";
  askStatus.classList.add("loading");

  if (!currentSessionId) {
    await newChat();
  }

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, session_id: currentSessionId })
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "request failed");
    }

    lastQuestion = q;
    lastAnswer = data.answer || "";
    lastQaId = data.qa_id || "";
    if (data.session_id) setStoredSessionId(data.session_id);

    document.getElementById("qa-pill").textContent = `qa_id: ${lastQaId || "-"}`;
    document.getElementById("trace-pill").textContent = `trace_id: ${data.trace_id || "-"}`;
    document.getElementById("answer").textContent = data.answer || "";
    document.getElementById("eval").textContent = JSON.stringify(data.evaluation || {}, null, 2);
    document.getElementById("similarity").textContent = JSON.stringify(data.similarity || {}, null, 2);
    document.getElementById("context").textContent = data.context || "";
    document.getElementById("chunks").textContent = JSON.stringify(data.chunks || [], null, 2);

    askStatus.textContent = "Done";
    setGlobalStatus("Request completed");
    await refreshSessions();
  } catch (e) {
    askStatus.textContent = "Failed";
    setGlobalStatus(`Ask failed: ${e}`, true);
  } finally {
    askStatus.classList.remove("loading");
  }
}

async function sendFeedback(rating) {
  if (!lastQuestion || !lastAnswer) {
    document.getElementById("fb-status").textContent = "Ask a question first.";
    return;
  }

  const res = await fetch("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      qa_id: lastQaId,
      question: lastQuestion,
      answer: lastAnswer,
      rating
    })
  });

  const data = await res.json();
  if (res.ok) {
    document.getElementById("fb-status").textContent = `Feedback saved (${rating}/5)`;
    setGlobalStatus("Feedback recorded");
    await loadSessionMessages(currentSessionId);
  } else {
    document.getElementById("fb-status").textContent = `Feedback failed: ${data.detail || "unknown"}`;
    setGlobalStatus("Feedback failed", true);
  }
}

refreshSessions();
