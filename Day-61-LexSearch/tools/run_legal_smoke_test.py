from __future__ import annotations

from pathlib import Path

import requests


def main() -> None:
    base = "http://127.0.0.1:8000"
    file_path = Path(__file__).resolve().parents[1] / "test-data" / "legal_sample_case.pdf"

    with file_path.open("rb") as handle:
        upload_res = requests.post(
            f"{base}/api/upload",
            files={"files": (file_path.name, handle, "application/pdf")},
            timeout=180,
        )

    print("UPLOAD_STATUS", upload_res.status_code)
    print(upload_res.text)

    payload = {
        "query": "breach of contract and liquidated damages under Section 74",
        "top_k": 6,
        "mode": "hybrid",
        "session_id": "demo_legal_case",
    }
    search_res = requests.post(f"{base}/api/search", json=payload, timeout=180)
    print("SEARCH_STATUS", search_res.status_code)
    print(search_res.text)

    if search_res.ok:
        data = search_res.json()
        fused = data.get("fused_results", [])
        if fused:
            top = fused[0]
            print("TOP_FILE", top.get("file_name"))
            print("TOP_SCORE", top.get("score"))
            print("TOP_SNIPPET", top.get("snippet", "")[:220])

    filtered_payload = {
        "query": "breach of contract and liquidated damages under Section 74",
        "top_k": 6,
        "mode": "hybrid",
        "session_id": "demo_legal_case",
        "file_name_contains": "legal_sample_case",
    }
    filtered_res = requests.post(f"{base}/api/search", json=filtered_payload, timeout=180)
    print("FILTERED_SEARCH_STATUS", filtered_res.status_code)
    print(filtered_res.text)


if __name__ == "__main__":
    main()
