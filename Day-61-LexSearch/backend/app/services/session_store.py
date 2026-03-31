from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List


class SessionStore:
    """In-memory per-session history and saved results."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: {"history": [], "saved": []}
        )

    def add_history(self, session_id: str, query: str, mode: str, top_k: int) -> None:
        if not session_id:
            return
        self._sessions[session_id]["history"].insert(
            0,
            {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "mode": mode,
                "top_k": top_k,
            },
        )
        self._sessions[session_id]["history"] = self._sessions[session_id]["history"][:50]

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self._sessions[session_id]["history"] if session_id in self._sessions else []

    def clear_history(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["history"] = []

    def add_saved(self, session_id: str, result: Dict[str, Any]) -> None:
        if not session_id:
            return
        self._sessions[session_id]["saved"].insert(
            0,
            {
                "timestamp": datetime.utcnow().isoformat(),
                "result": result,
            },
        )
        self._sessions[session_id]["saved"] = self._sessions[session_id]["saved"][:200]

    def get_saved(self, session_id: str) -> List[Dict[str, Any]]:
        return self._sessions[session_id]["saved"] if session_id in self._sessions else []
