const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function parseResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const msg = data?.detail || data?.error || `Request failed (${response.status})`;
    throw new Error(msg);
  }
  return data;
}

export async function fetchStatus() {
  const response = await fetch(`${API_BASE}/api/status`);
  return parseResponse(response);
}

export async function uploadDocuments(files) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const response = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData,
  });
  return parseResponse(response);
}

export async function searchDocuments(payload) {
  const response = await fetch(`${API_BASE}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function fetchHistory(sessionId) {
  const response = await fetch(`${API_BASE}/api/session/${sessionId}/history`);
  return parseResponse(response);
}

export async function clearHistory(sessionId) {
  const response = await fetch(`${API_BASE}/api/session/${sessionId}/history`, {
    method: "DELETE",
  });
  return parseResponse(response);
}

export async function fetchSaved(sessionId) {
  const response = await fetch(`${API_BASE}/api/session/${sessionId}/saved`);
  return parseResponse(response);
}

export async function saveResult(sessionId, result) {
  const response = await fetch(`${API_BASE}/api/session/${sessionId}/saved`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ result }),
  });
  return parseResponse(response);
}
