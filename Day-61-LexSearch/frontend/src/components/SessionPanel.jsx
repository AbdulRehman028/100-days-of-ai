import { useMemo, useState } from "react";

export default function SessionPanel({ history, saved, onSelectSaved, onClearHistory }) {
  const [tab, setTab] = useState("history");
  const items = useMemo(() => (tab === "history" ? history : saved), [tab, history, saved]);

  return (
    <section className="glass p-4 md:p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-display text-lg">Session Memory</h3>
        <div className="flex items-center gap-1 bg-white/5 border border-white/10 rounded-lg p-1 text-xs">
          <button
            type="button"
            onClick={() => setTab("history")}
            className={`px-2 py-1 rounded ${tab === "history" ? "bg-pulse/20 text-pulse" : "text-slate-300"}`}
          >
            History ({history.length})
          </button>
          <button
            type="button"
            onClick={() => setTab("saved")}
            className={`px-2 py-1 rounded ${tab === "saved" ? "bg-mint/20 text-mint" : "text-slate-300"}`}
          >
            Saved ({saved.length})
          </button>
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-sm text-slate-300 font-semibold">{tab === "history" ? "Recent Queries" : "Pinned Results"}</h4>
          {tab === "history" && (
            <button className="text-xs px-2 py-1 rounded bg-white/5 border border-white/10" onClick={onClearHistory}>
              Clear
            </button>
          )}
        </div>
        <div className="space-y-2 max-h-52 overflow-auto pr-1 custom-scroll">
          {items.length === 0 && <p className="text-sm text-slate-500">No {tab} yet.</p>}

          {tab === "history" &&
            history.map((item, idx) => (
              <div key={`${item.timestamp}_${idx}`} className="soft-card p-2 text-xs">
                <p className="text-slate-200 truncate">{item.query}</p>
                <p className="text-slate-500 mt-1">{item.mode} · top {item.top_k}</p>
              </div>
            ))}

          {tab === "saved" &&
            saved.map((item, idx) => (
              <button
                key={`${item.timestamp}_${idx}`}
                className="soft-card p-2 text-left w-full hover:border-mint/40"
                onClick={() => onSelectSaved(item.result)}
                type="button"
              >
                <p className="text-sm text-slate-200 truncate">{item.result.file_name}</p>
                <p className="text-xs text-slate-500 truncate">{item.result.snippet}</p>
              </button>
            ))}
        </div>
      </div>
    </section>
  );
}
