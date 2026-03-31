export default function SearchPanel({
  query,
  fileFilter,
  mode,
  topK,
  searching,
  onChange,
  onSearch,
}) {
  const modes = [
    { value: "hybrid", label: "Hybrid (RRF)" },
    { value: "dense", label: "Semantic (FAISS)" },
    { value: "sparse", label: "Keyword (BM25)" },
  ];

  const topKOptions = [5, 8, 10, 12, 15, 20];

  return (
    <section className="glass p-4 md:p-5 space-y-4">
      <div>
        <h3 className="font-display text-lg">Retrieval Query</h3>
        <p className="text-xs text-slate-400 mt-1">Use natural legal language, facts, statutes, and outcome terms.</p>
      </div>

      <textarea
        value={query}
        onChange={(e) => onChange({ query: e.target.value })}
        onKeyDown={(e) => {
          if ((e.ctrlKey || e.metaKey) && e.key === "Enter") onSearch();
        }}
        className="w-full min-h-[110px] bg-steel/90 border border-white/10 rounded-xl p-3 text-sm md:text-[15px] outline-none focus:border-pulse focus:ring-2 focus:ring-pulse/20"
        placeholder="Search legal concepts, facts, statutes, or precedent language..."
      />
      <p className="text-[11px] text-slate-500 -mt-1">Tip: press Ctrl/Cmd + Enter to run search</p>

      <div>
        <p className="text-xs text-slate-400 mb-1">Optional filename filter</p>
        <input
          value={fileFilter}
          onChange={(e) => onChange({ fileFilter: e.target.value })}
          className="w-full bg-steel/90 border border-white/10 rounded-xl p-2.5 text-sm outline-none focus:border-mint focus:ring-2 focus:ring-mint/20"
          placeholder="Example: legal_sample_case"
        />
      </div>

      <div className="space-y-2">
        <p className="text-xs text-slate-400">Search Mode</p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {modes.map((m) => (
            <button
              key={m.value}
              type="button"
              onClick={() => onChange({ mode: m.value })}
              className={`rounded-lg px-3 py-2 text-sm border transition-all ${
                mode === m.value
                  ? "bg-pulse/20 text-pulse border-pulse/50"
                  : "bg-steel/70 text-slate-300 border-white/10 hover:border-white/20"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div>
          <p className="text-xs text-slate-400 mb-1">Top K per ranking</p>
          <div className="flex flex-wrap gap-2">
            {topKOptions.map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => onChange({ topK: n })}
                className={`px-2.5 py-1.5 rounded-md text-xs border ${
                  topK === n
                    ? "bg-mint/20 text-mint border-mint/50"
                    : "bg-steel/80 text-slate-300 border-white/10"
                }`}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        <div>
          <p className="text-xs text-slate-400 mb-1">Preset prompts</p>
          <div className="flex flex-wrap gap-2">
            {[
              "contract breach remedy",
              "anticipatory breach precedent",
              "liquidated damages enforceability",
            ].map((preset) => (
              <button
                key={preset}
                type="button"
                onClick={() => onChange({ query: preset })}
                className="px-2.5 py-1.5 rounded-md text-[11px] border border-white/10 bg-white/5 text-slate-300 hover:border-pulse/40"
              >
                {preset}
              </button>
            ))}
          </div>
        </div>
      </div>

      <button
        className="w-full py-2.5 rounded-xl bg-gradient-to-r from-pulse via-sky-300 to-mint text-abyss font-semibold disabled:opacity-60 transition-transform hover:-translate-y-0.5"
        onClick={onSearch}
        disabled={searching || !query.trim()}
      >
        {searching ? "Searching..." : "Run Retrieval"}
      </button>
    </section>
  );
}
