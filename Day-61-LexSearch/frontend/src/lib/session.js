export function getOrCreateSessionId() {
  const key = "lexsearch_session_id";
  const existing = localStorage.getItem(key);
  if (existing) return existing;

  const random = Math.random().toString(36).slice(2, 10);
  const sid = `sess_${Date.now()}_${random}`;
  localStorage.setItem(key, sid);
  return sid;
}
