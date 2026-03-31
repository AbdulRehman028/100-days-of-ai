import { useRef, useState } from "react";

export default function UploadPanel({ uploading, onUpload, uploadStats, warnings }) {
  const fileInputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handlePickedFiles = async (files) => {
    if (!files.length) return;
    await onUpload(files);
  };

  const handleFiles = async (event) => {
    const files = Array.from(event.target.files || []);
    await handlePickedFiles(files);
    event.target.value = "";
  };

  const onDrop = async (event) => {
    event.preventDefault();
    setDragging(false);
    const files = Array.from(event.dataTransfer.files || []).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    await handlePickedFiles(files);
  };

  return (
    <section className="glass p-4 md:p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-display text-lg">Document Ingestion</h3>
        <span className="text-xs text-slate-400">PDF + OCR fallback</span>
      </div>

      <div
        className={`rounded-xl border border-dashed p-4 text-center transition-colors ${
          dragging ? "border-pulse bg-pulse/10" : "border-white/20 bg-white/5"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <p className="text-sm text-slate-300">Drop PDF files here</p>
        <p className="text-xs text-slate-500 mt-1">or</p>
        <button
          className="mt-2 px-4 py-2.5 rounded-lg bg-gradient-to-r from-pulse to-mint text-abyss font-semibold disabled:opacity-60"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
          type="button"
        >
          {uploading ? "Uploading..." : "Choose Files"}
        </button>
      </div>

      <input ref={fileInputRef} type="file" accept=".pdf" multiple className="hidden" onChange={handleFiles} />

      <div className="grid grid-cols-3 gap-2 text-center text-xs">
        <div className="soft-card p-2">
          <p className="text-slate-400">Files</p>
          <p className="text-base font-mono text-pulse">{uploadStats.uploaded_files ?? 0}</p>
        </div>
        <div className="soft-card p-2">
          <p className="text-slate-400">Docs</p>
          <p className="text-base font-mono text-mint">{uploadStats.processed_documents ?? 0}</p>
        </div>
        <div className="soft-card p-2">
          <p className="text-slate-400">Chunks</p>
          <p className="text-base font-mono text-amber">{uploadStats.created_chunks ?? 0}</p>
        </div>
      </div>

      {warnings?.length > 0 && (
        <div className="soft-card p-3 text-xs text-amber space-y-1 max-h-28 overflow-auto">
          {warnings.map((w, idx) => (
            <p key={idx}>• {w}</p>
          ))}
        </div>
      )}
    </section>
  );
}
