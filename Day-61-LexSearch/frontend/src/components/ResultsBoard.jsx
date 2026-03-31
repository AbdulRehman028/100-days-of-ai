import ResultCard from "./ResultCard";

function Section({ title, subtitle, items, onSelect, onSave, compact = false }) {
  return (
    <section className="glass p-4 md:p-5">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="font-display text-lg">{title}</h3>
          {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
        </div>
        <span className="text-xs text-slate-400">{items.length} hits</span>
      </div>

      <div className="space-y-3 max-h-[460px] overflow-auto pr-1 custom-scroll">
        {items.length === 0 && <p className="text-sm text-slate-500">No results for this ranking.</p>}
        {items.map((item) => (
          <ResultCard
            key={item.chunk_id + title}
            result={item}
            onSelect={onSelect}
            onSave={onSave}
            compact={compact}
          />
        ))}
      </div>
    </section>
  );
}

export default function ResultsBoard({ results, onSelect, onSave }) {
  const fused = results.fused_results || [];
  const dense = results.dense_results || [];
  const sparse = results.sparse_results || [];

  return (
    <div className="space-y-4">
      <Section
        title="Fused Ranking (RRF)"
        subtitle="Primary decision list blended from semantic and keyword ranks"
        items={fused}
        onSelect={onSelect}
        onSave={onSave}
      />

      <div className="grid grid-cols-1 2xl:grid-cols-2 gap-4">
        <Section
          title="Semantic Vector Search (FAISS)"
          subtitle="Contextual similarity over embedding space"
          items={dense}
          onSelect={onSelect}
          onSave={onSave}
          compact
        />
        <Section
          title="Keyword Search (BM25)"
          subtitle="Exact term matching and lexical weighting"
          items={sparse}
          onSelect={onSelect}
          onSave={onSave}
          compact
        />
      </div>
    </div>
  );
}
