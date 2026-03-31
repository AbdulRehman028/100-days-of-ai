export default function MetadataSidebar({ selected, onSave }) {
  return (
    <aside className="glass p-4 md:p-5 h-fit sticky top-4">
      <h3 className="font-display text-lg mb-3">Case Metadata</h3>
      {!selected && (
        <div className="soft-card p-4 text-center">
          <p className="text-3xl mb-2">🗂️</p>
          <p className="text-sm text-slate-400">Select a result to inspect extracted legal metadata.</p>
        </div>
      )}

      {selected && (
        <div className="space-y-3">
          <div className="soft-card p-3 border-pulse/30">
            <p className="text-xs text-slate-400 uppercase tracking-wide">Document</p>
            <p className="font-semibold text-sm mt-1 leading-relaxed break-words">{selected.file_name}</p>
            <p className="text-xs text-slate-500 mt-1">chunk {selected.chunk_id}</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 2xl:grid-cols-1 gap-2 text-sm">
            <div className="soft-card p-3">
              <p className="text-xs text-slate-400">Case Date</p>
              <p className="mt-1">{selected.metadata?.case_date || "Unknown"}</p>
            </div>
            <div className="soft-card p-3">
              <p className="text-xs text-slate-400">Court</p>
              <p className="mt-1">{selected.metadata?.court || "Unknown"}</p>
            </div>
            <div className="soft-card p-3">
              <p className="text-xs text-slate-400">Parties</p>
              <p className="mt-1 break-words">{selected.metadata?.parties || "Unknown"}</p>
            </div>
          </div>

          <div className="soft-card p-3">
            <p className="text-xs text-slate-400 mb-1 uppercase tracking-wide">Snippet Preview</p>
            <p className="text-sm leading-relaxed text-slate-200">{selected.snippet}</p>
          </div>

          <button
            className="w-full py-2.5 rounded-lg bg-gradient-to-r from-mint/30 to-pulse/30 border border-mint/40 text-sm hover:border-pulse/50"
            onClick={() => onSave(selected)}
          >
            Save Result to Session
          </button>
        </div>
      )}
    </aside>
  );
}
