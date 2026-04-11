import { useEffect, useMemo, useReducer, useRef } from "react";
import MetadataSidebar from "./components/MetadataSidebar";
import ResultsBoard from "./components/ResultsBoard";
import SearchPanel from "./components/SearchPanel";
import SessionPanel from "./components/SessionPanel";
import UploadPanel from "./components/UploadPanel";
import {
  clearHistory,
  fetchHistory,
  fetchSaved,
  fetchStatus,
  saveResult,
  searchDocuments,
  uploadDocuments,
} from "./lib/api";
import { getOrCreateSessionId } from "./lib/session";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const initialState = {
  sessionId: getOrCreateSessionId(),
  status: { documents: 0, chunks: 0, embedding_model: "-" },
  uploadStats: { uploaded_files: 0, processed_documents: 0, created_chunks: 0 },
  warnings: [],
  uploading: false,
  query: "breach of contract damages under commercial lease",
  fileFilter: "",
  mode: "hybrid",
  topK: 8,
  searching: false,
  results: { dense_results: [], sparse_results: [], fused_results: [] },
  selectedResult: null,
  history: [],
  saved: [],
  toast: null,
  error: null,
};

function reducer(state, action) {
  switch (action.type) {
    case "SET_STATUS":
      return { ...state, status: action.payload };
    case "SET_UPLOADING":
      return { ...state, uploading: action.payload };
    case "SET_UPLOAD_RESULT":
      return {
        ...state,
        uploadStats: action.payload,
        warnings: action.payload.warnings || [],
      };
    case "SET_SEARCH_FORM":
      return { ...state, ...action.payload };
    case "SET_SEARCHING":
      return { ...state, searching: action.payload };
    case "SET_RESULTS":
      return { ...state, results: action.payload };
    case "SET_SELECTED":
      return { ...state, selectedResult: action.payload };
    case "SET_HISTORY":
      return { ...state, history: action.payload };
    case "SET_SAVED":
      return { ...state, saved: action.payload };
    case "SET_TOAST":
      return { ...state, toast: action.payload };
    case "SET_ERROR":
      return { ...state, error: action.payload };
    default:
      return state;
  }
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const toastTimeout = useRef(null);

  const totalHits = useMemo(() => {
    return {
      fused: state.results.fused_results?.length || 0,
      dense: state.results.dense_results?.length || 0,
      sparse: state.results.sparse_results?.length || 0,
    };
  }, [state.results]);

  const pushToast = (message, kind = "success") => {
    dispatch({ type: "SET_TOAST", payload: { message, kind } });
    if (toastTimeout.current) clearTimeout(toastTimeout.current);
    toastTimeout.current = setTimeout(() => {
      dispatch({ type: "SET_TOAST", payload: null });
    }, 2600);
  };

  const loadStatus = async () => {
    const data = await fetchStatus();
    dispatch({ type: "SET_STATUS", payload: data });
  };

  const loadSession = async () => {
    const [historyData, savedData] = await Promise.all([
      fetchHistory(state.sessionId),
      fetchSaved(state.sessionId),
    ]);
    dispatch({ type: "SET_HISTORY", payload: historyData.items || [] });
    dispatch({ type: "SET_SAVED", payload: savedData.items || [] });
  };

  useEffect(() => {
    (async () => {
      try {
        await loadStatus();
        await loadSession();
      } catch (error) {
        dispatch({ type: "SET_ERROR", payload: error.message });
        pushToast(error.message, "error");
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleUpload = async (files) => {
    dispatch({ type: "SET_UPLOADING", payload: true });
    dispatch({ type: "SET_ERROR", payload: null });
    try {
      const data = await uploadDocuments(files);
      dispatch({ type: "SET_UPLOAD_RESULT", payload: data });
      await loadStatus();
      pushToast(`Indexed ${data.created_chunks} chunks from ${data.processed_documents} docs`);
    } catch (error) {
      dispatch({ type: "SET_ERROR", payload: error.message });
      pushToast(error.message, "error");
    } finally {
      dispatch({ type: "SET_UPLOADING", payload: false });
    }
  };

  const handleSearch = async () => {
    if (!state.query.trim()) return;

    dispatch({ type: "SET_SEARCHING", payload: true });
    dispatch({ type: "SET_ERROR", payload: null });
    try {
      const data = await searchDocuments({
        query: state.query,
        mode: state.mode,
        top_k: state.topK,
        session_id: state.sessionId,
        file_name_contains: state.fileFilter?.trim() || null,
      });

      dispatch({ type: "SET_RESULTS", payload: data });

      const fallback = data.fused_results?.[0] || data.dense_results?.[0] || data.sparse_results?.[0] || null;
      dispatch({ type: "SET_SELECTED", payload: fallback });

      const historyData = await fetchHistory(state.sessionId);
      dispatch({ type: "SET_HISTORY", payload: historyData.items || [] });

      pushToast(`Search completed: ${data.fused_results.length} fused hits`);
    } catch (error) {
      dispatch({ type: "SET_ERROR", payload: error.message });
      pushToast(error.message, "error");
    } finally {
      dispatch({ type: "SET_SEARCHING", payload: false });
    }
  };

  const handleSaveResult = async (result) => {
    try {
      await saveResult(state.sessionId, result);
      const data = await fetchSaved(state.sessionId);
      dispatch({ type: "SET_SAVED", payload: data.items || [] });
      pushToast("Result saved to session");
    } catch (error) {
      pushToast(error.message, "error");
    }
  };

  const handleClearHistory = async () => {
    try {
      await clearHistory(state.sessionId);
      dispatch({ type: "SET_HISTORY", payload: [] });
      pushToast("History cleared");
    } catch (error) {
      pushToast(error.message, "error");
    }
  };

  return (
    <div className="min-h-screen relative pb-10">
      {state.toast && (
        <div
          className={`fixed top-4 right-4 z-50 px-4 py-2.5 rounded-xl border text-sm backdrop-blur-md shadow-neon ${
            state.toast.kind === "error"
              ? "bg-rose/20 border-rose/50 text-rose"
              : "bg-mint/20 border-mint/50 text-mint"
          }`}
        >
          {state.toast.message}
        </div>
      )}

      <header className="max-w-[1720px] mx-auto px-4 md:px-6 pt-6 pb-5">
        <div className="glass p-5 md:p-6 overflow-hidden relative">
          <div className="absolute -top-20 -right-16 w-52 h-52 rounded-full bg-pulse/20 blur-3xl pointer-events-none" />
          <div className="absolute -bottom-20 -left-16 w-56 h-56 rounded-full bg-mint/20 blur-3xl pointer-events-none" />

          <div className="relative flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.28em] text-pulse font-mono">Day 61 • Legal Retrieval Workspace</p>
              <h1 className="text-3xl md:text-4xl xl:text-5xl font-display font-extrabold mt-2 leading-tight">
                LexSearch
                <span className="text-slate-400 text-xl md:text-2xl xl:text-3xl font-semibold">  Document Intelligence Portal</span>
              </h1>
              <p className="text-slate-300/90 mt-3 text-sm md:text-base max-w-3xl">
                Compare semantic retrieval and keyword retrieval side-by-side, inspect legal metadata instantly, and save strategic findings by session.
              </p>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs min-w-full lg:min-w-[620px]">
              <div className="kpi-card">
                <p className="kpi-label">Docs</p>
                <p className="kpi-value">{state.status.documents}</p>
              </div>
              <div className="kpi-card">
                <p className="kpi-label">Chunks</p>
                <p className="kpi-value">{state.status.chunks}</p>
              </div>
              <div className="kpi-card">
                <p className="kpi-label">Fused Hits</p>
                <p className="kpi-value">{totalHits.fused}</p>
              </div>
              <div className="kpi-card col-span-2 md:col-span-1">
                <p className="kpi-label">Embed Mode</p>
                <p className="kpi-value text-[11px] uppercase tracking-wide">{state.status.embedding_mode || "-"}</p>
              </div>
              <a
                href={`${API_BASE}/docs`}
                target="_blank"
                rel="noreferrer"
                className="kpi-card hover:border-pulse transition-colors col-span-2 md:col-span-1"
              >
                <p className="kpi-label">API Docs</p>
                <p className="kpi-value text-[11px]">Open /docs</p>
              </a>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-[1720px] mx-auto px-4 md:px-6 pb-8 grid grid-cols-1 2xl:grid-cols-[350px_minmax(0,1fr)_340px] xl:grid-cols-[350px_minmax(0,1fr)] gap-4 xl:gap-5">
        <div className="space-y-4 xl:sticky xl:top-4 h-fit">
          <UploadPanel
            uploading={state.uploading}
            onUpload={handleUpload}
            uploadStats={state.uploadStats}
            warnings={state.warnings}
          />

          <SearchPanel
            query={state.query}
            fileFilter={state.fileFilter}
            mode={state.mode}
            topK={state.topK}
            searching={state.searching}
            onChange={(payload) => dispatch({ type: "SET_SEARCH_FORM", payload })}
            onSearch={handleSearch}
          />

          <SessionPanel
            history={state.history}
            saved={state.saved}
            onSelectSaved={(r) => dispatch({ type: "SET_SELECTED", payload: r })}
            onClearHistory={handleClearHistory}
          />

          {state.error && (
            <div className="glass p-3 border-rose/40 text-sm text-rose">{state.error}</div>
          )}
        </div>

        <div className="space-y-4">
          <section className="soft-card px-4 py-3 flex flex-wrap items-center justify-between gap-3 text-xs md:text-sm">
            <div className="flex flex-wrap items-center gap-2">
              <span className="px-2 py-1 rounded-full bg-pulse/20 text-pulse border border-pulse/30 font-mono uppercase tracking-wide">{state.mode}</span>
              <span className="text-slate-400">Top {state.topK} per ranking</span>
              {state.fileFilter?.trim() && <span className="text-slate-400">filter {state.fileFilter.trim()}</span>}
              <span className="text-slate-500">session {state.sessionId.slice(0, 18)}...</span>
            </div>
            <div className="flex items-center gap-2 text-slate-400">
              <span>Dense: {totalHits.dense}</span>
              <span>•</span>
              <span>Sparse: {totalHits.sparse}</span>
            </div>
          </section>

          {(state.results.fused_results?.length || state.results.dense_results?.length || state.results.sparse_results?.length) ? (
            <ResultsBoard
              results={state.results}
              onSelect={(result) => dispatch({ type: "SET_SELECTED", payload: result })}
              onSave={handleSaveResult}
            />
          ) : (
            <div className="glass p-8 text-center">
              <p className="text-5xl mb-3">⚖️</p>
              <h3 className="text-xl font-display">Upload legal documents and run retrieval</h3>
              <p className="text-sm text-slate-400 mt-2">
                Compare semantic FAISS retrieval against BM25 keyword ranking, then inspect fused RRF results.
              </p>
            </div>
          )}
        </div>

        <div className="xl:col-span-2 2xl:col-span-1">
          <MetadataSidebar selected={state.selectedResult} onSave={handleSaveResult} />
        </div>
      </main>
    </div>
  );
}