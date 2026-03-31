export default function ResultCard({ result, onSelect, onSave, compact = false }) {
  return (
    <article className="soft-card p-3.5 space-y-3 border-white/10 hover:border-pulse/35 transition-colors">
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-pulse/20 border border-pulse/35 text-pulse">
              rank #{result.rank}
            </span>
            <span className="text-[10px] font-mono text-slate-500">score {result.score.toFixed(4)}</span>
          </div>
          <p className="text-xs text-slate-300 break-all">{result.file_name}</p>
        </div>

        <div className="flex gap-2">
          <button
            className="text-[11px] px-2 py-1 rounded bg-white/5 border border-white/10 hover:border-pulse/40"
            onClick={() => onSelect(result)}
            type="button"
          >
            Meta
          </button>
          {!compact && (
            <button
              className="text-[11px] px-2 py-1 rounded bg-mint/25 border border-mint/40 hover:border-mint/70"
              onClick={() => onSave(result)}
              type="button"
            >
              Save
            </button>
          )}
        </div>
      </div>

      <div className="marked-snippet text-sm leading-relaxed text-slate-200" dangerouslySetInnerHTML={{ __html: result.highlighted_snippet }} />
    </article>
  );
}
