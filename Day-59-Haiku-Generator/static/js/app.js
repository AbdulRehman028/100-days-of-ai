/* ══════════════════════════════════════════════
   Haiku Generator — App Controller
   Day 59 · Proper state management pattern
   ══════════════════════════════════════════════ */

(() => {
    "use strict";

    // ─── State Store ──────────────────────────
    const state = {
        // Engine
        model: null,
        device: null,
        moods: {},
        seasons: {},
        themes: [],

        // Selection
        selectedMood: "serene",
        selectedSeason: "none",

        // Generation
        isGenerating: false,
        lastResult: null,
        loadingTimerRef: null,

        // Stats
        totalHaikus: 0,
    };

    // ─── DOM Cache ────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const dom = {};
    function cacheDom() {
        dom.themeInput     = $("#themeInput");
        dom.charCount      = $("#charCount");
        dom.moodGrid       = $("#moodGrid");
        dom.seasonPills    = $("#seasonPills");
        dom.themeChips     = $("#themeChips");
        dom.countSelect    = $("#countSelect");
        dom.tempSlider     = $("#tempSlider");
        dom.tempValue      = $("#tempValue");
        dom.generateBtn    = $("#generateBtn");
        dom.collectionList = $("#collectionList");
        dom.loadingOverlay = $("#loadingOverlay");
        dom.loadingStep    = $("#loadingStep");
        dom.loadingTimer   = $("#loadingTimer");
        dom.toast          = $("#toast");
        dom.emptyState     = $("#emptyState");
        dom.haikuDisplay   = $("#haikuDisplay");
        dom.haikusContainer= $("#haikusContainer");
        dom.resultTitle    = $("#resultTitle");
        dom.resultMeta     = $("#resultMeta");
        dom.analysisCard   = $("#analysisCard");
        dom.analysisContent= $("#analysisContent");
        dom.promptCard     = $("#promptCard");
        dom.promptSystem   = $("#promptSystem");
        dom.promptUser     = $("#promptUser");
        dom.techniquesTags = $("#techniquesTags");
        dom.statsCard      = $("#statsCard");
        dom.statsGrid      = $("#statsGrid");
        dom.engineModel    = $("#engineModel");
        dom.engineDevice   = $("#engineDevice");
        dom.totalHaikus    = $("#totalHaikus");
        dom.particles      = $("#particles");
    }

    // ─── Helpers ──────────────────────────────
    function esc(str) {
        const d = document.createElement("div");
        d.textContent = str;
        return d.innerHTML;
    }

    function showToast(message, type = "success") {
        dom.toast.textContent = message;
        dom.toast.className = `toast ${type} show`;
        setTimeout(() => dom.toast.classList.remove("show"), 2800);
    }

    function setLoading(on) {
        state.isGenerating = on;
        dom.generateBtn.disabled = on;
        if (on) {
            dom.loadingOverlay.classList.add("active");
            const steps = [
                "Building prompt...",
                "Generating haiku...",
                "Composing lines...",
                "Counting syllables...",
                "Selecting best haiku...",
            ];
            let idx = 0;
            const start = performance.now();
            dom.loadingStep.textContent = steps[0];

            state.loadingTimerRef = setInterval(() => {
                const elapsed = ((performance.now() - start) / 1000).toFixed(1);
                dom.loadingTimer.textContent = elapsed + "s";

                // Rotate messages every 5s
                if (parseFloat(elapsed) > (idx + 1) * 5 && idx < steps.length - 1) {
                    idx++;
                    dom.loadingStep.textContent = steps[idx];
                }
            }, 100);
        } else {
            dom.loadingOverlay.classList.remove("active");
            if (state.loadingTimerRef) {
                clearInterval(state.loadingTimerRef);
                state.loadingTimerRef = null;
            }
        }
    }

    // ─── Particle System ──────────────────────
    function initParticles() {
        const count = 25;
        for (let i = 0; i < count; i++) {
            const p = document.createElement("div");
            p.className = "particle";
            p.style.left = Math.random() * 100 + "%";
            p.style.animationDuration = (12 + Math.random() * 18) + "s";
            p.style.animationDelay = (Math.random() * 15) + "s";
            p.style.width = (2 + Math.random() * 3) + "px";
            p.style.height = p.style.width;
            p.style.opacity = 0.1 + Math.random() * 0.2;
            dom.particles.appendChild(p);
        }
    }

    // ─── Build UI Components ──────────────────
    function buildMoodGrid() {
        dom.moodGrid.innerHTML = "";
        for (const [key, m] of Object.entries(state.moods)) {
            const chip = document.createElement("button");
            chip.className = "mood-chip" + (key === state.selectedMood ? " active" : "");
            chip.dataset.mood = key;
            chip.innerHTML = `
                <span class="mood-icon">${m.icon}</span>
                <span class="mood-name">${esc(m.label)}</span>
                <span class="mood-desc">${esc(m.desc)}</span>
            `;
            chip.addEventListener("click", () => selectMood(key));
            dom.moodGrid.appendChild(chip);
        }
    }

    function selectMood(key) {
        state.selectedMood = key;
        dom.moodGrid.querySelectorAll(".mood-chip").forEach(c => {
            c.classList.toggle("active", c.dataset.mood === key);
        });
    }

    function buildSeasonPills() {
        dom.seasonPills.innerHTML = "";
        for (const [key, s] of Object.entries(state.seasons)) {
            const pill = document.createElement("button");
            pill.className = "season-pill" + (key === state.selectedSeason ? " active" : "");
            pill.dataset.season = key;
            pill.innerHTML = `${s.icon} ${esc(s.label)}`;
            pill.addEventListener("click", () => selectSeason(key));
            dom.seasonPills.appendChild(pill);
        }
    }

    function selectSeason(key) {
        state.selectedSeason = key;
        dom.seasonPills.querySelectorAll(".season-pill").forEach(p => {
            p.classList.toggle("active", p.dataset.season === key);
        });
    }

    function buildThemeChips() {
        dom.themeChips.innerHTML = "";
        // Show a random subset of 10
        const shuffled = [...state.themes].sort(() => Math.random() - 0.5).slice(0, 10);
        for (const theme of shuffled) {
            const chip = document.createElement("button");
            chip.className = "theme-chip";
            chip.textContent = theme;
            chip.addEventListener("click", () => {
                dom.themeInput.value = theme;
                updateCharCount();
            });
            dom.themeChips.appendChild(chip);
        }
    }

    // ─── Input Bindings ───────────────────────
    function bindInputs() {
        // Char count
        dom.themeInput.addEventListener("input", updateCharCount);

        // Temperature slider
        dom.tempSlider.addEventListener("input", () => {
            dom.tempValue.textContent = parseFloat(dom.tempSlider.value).toFixed(2);
        });

        // Enter key generates
        dom.themeInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !state.isGenerating) generate();
        });
    }

    function updateCharCount() {
        const len = dom.themeInput.value.length;
        dom.charCount.textContent = `${len}/80`;
    }

    // ─── API Calls ────────────────────────────
    async function fetchStatus() {
        try {
            const res = await fetch("/status");
            const data = await res.json();
            state.model = data.model;
            state.device = data.device;
            state.moods = data.moods;
            state.seasons = data.seasons;
            state.totalHaikus = data.haikus_generated;

            dom.engineModel.textContent = data.model || "Unknown";
            dom.engineDevice.textContent = data.device || "—";
            dom.totalHaikus.textContent = state.totalHaikus;

            buildMoodGrid();
            buildSeasonPills();
        } catch (e) {
            dom.engineModel.textContent = "Error loading";
            console.error("Status fetch failed:", e);
        }
    }

    async function fetchThemes() {
        try {
            const res = await fetch("/themes");
            const data = await res.json();
            state.themes = data.themes || [];
            buildThemeChips();
        } catch (e) {
            console.error("Themes fetch failed:", e);
        }
    }

    async function fetchCollection() {
        try {
            const res = await fetch("/collection");
            const data = await res.json();
            displayCollection(data.collection || []);
        } catch (e) {
            console.error("Collection fetch failed:", e);
        }
    }

    // ─── Generate ─────────────────────────────
    async function generate() {
        const theme = dom.themeInput.value.trim();
        if (!theme) {
            showToast("Please enter a theme first.", "error");
            dom.themeInput.focus();
            return;
        }
        if (state.isGenerating) return;

        setLoading(true);

        try {
            const res = await fetch("/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    theme,
                    mood: state.selectedMood,
                    season: state.selectedSeason,
                    temperature: parseFloat(dom.tempSlider.value),
                    count: parseInt(dom.countSelect.value),
                }),
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({ error: "Generation failed" }));
                throw new Error(err.error || `HTTP ${res.status}`);
            }

            const data = await res.json();
            state.lastResult = data;
            state.totalHaikus += data.meta.count_generated;
            dom.totalHaikus.textContent = state.totalHaikus;

            displayHaikus(data);
            displayAnalysis(data);
            displayPromptInfo(data);
            displayStats(data);
            fetchCollection();

            showToast(
                `${data.meta.count_generated} haiku${data.meta.count_generated > 1 ? "s" : ""} generated in ${data.meta.time_seconds}s`,
                "success"
            );
        } catch (e) {
            showToast(e.message, "error");
            console.error("Generation error:", e);
        } finally {
            setLoading(false);
        }
    }

    // ─── Display: Haikus ──────────────────────
    function displayHaikus(data) {
        dom.emptyState.classList.add("hidden");
        dom.haikuDisplay.classList.remove("hidden");

        dom.resultTitle.textContent = `${data.meta.mood_icon} ${data.meta.theme}`;
        dom.resultMeta.textContent =
            `${data.meta.mood_label} · ${data.meta.season_label} · ${data.meta.count_generated} haiku${data.meta.count_generated > 1 ? "s" : ""} · ${data.meta.time_seconds}s`;

        dom.haikusContainer.innerHTML = "";
        data.haikus.forEach((h, i) => {
            const block = document.createElement("div");
            block.className = "haiku-block";
            block.dataset.num = String(i + 1).padStart(2, "0");
            block.style.animationDelay = `${i * 0.15}s`;

            const linesHtml = h.lines.map(l => `<div class="haiku-line">${esc(l)}</div>`).join("");

            const badgesHtml = h.syllables.map((s, j) => {
                const target = [5, 7, 5][j];
                let cls = "syl-na";
                if (target !== undefined) {
                    cls = s === target ? "syl-valid" : "syl-invalid";
                }
                return `<span class="syl-badge ${cls}">${s}/${target || "?"}</span>`;
            }).join("");

            block.innerHTML = `
                ${linesHtml}
                <div class="haiku-syllables">${badgesHtml}</div>
                <div class="haiku-score">${h.score} pts</div>
                <button class="haiku-copy-btn" onclick="window.HaikuApp.copySingle(${i})">copy</button>
            `;

            dom.haikusContainer.appendChild(block);
        });
    }

    // ─── Display: Syllable Analysis ───────────
    function displayAnalysis(data) {
        dom.analysisCard.classList.remove("hidden");

        const rows = data.haikus.map((h, i) => {
            const badge = h.is_valid_575
                ? '<span class="text-bamboo text-xs font-semibold">✓ Valid 5-7-5</span>'
                : '<span class="text-accent-light text-xs font-semibold">✗ Non-standard</span>';

            const syllableDetail = h.lines.map((line, j) => {
                const target = [5, 7, 5][j];
                const actual = h.syllables[j] || 0;
                const match = actual === target;
                return `
                    <div class="flex items-center gap-2 text-xs text-ink-300">
                        <span class="w-5 font-mono text-ink-500">L${j + 1}</span>
                        <span class="flex-1 font-serif italic">${esc(line)}</span>
                        <span class="font-mono ${match ? 'text-bamboo' : 'text-accent-light'}">${actual}/${target}</span>
                    </div>
                `;
            }).join("");

            return `
                <div class="mb-4 p-3 rounded-xl bg-white/[0.02] border border-white/[0.04]">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-xs text-ink-500 font-mono">Haiku #${i + 1}</span>
                        ${badge}
                    </div>
                    <div class="space-y-1">${syllableDetail}</div>
                </div>
            `;
        }).join("");

        dom.analysisContent.innerHTML = rows;
    }

    // ─── Display: Prompt Engineering ──────────
    function displayPromptInfo(data) {
        const pi = data.prompt_info;
        dom.promptCard.classList.remove("hidden");
        dom.promptSystem.textContent = pi.system_prompt;
        dom.promptUser.textContent = pi.user_prompt;

        dom.techniquesTags.innerHTML = pi.techniques
            .map(t => `<span class="tech-tag">${esc(t)}</span>`)
            .join("");
    }

    // ─── Display: Stats ───────────────────────
    function displayStats(data) {
        dom.statsCard.classList.remove("hidden");
        const m = data.meta;

        const stats = [
            { label: "Time", val: m.time_seconds + "s" },
            { label: "Attempts", val: m.attempts },
            { label: "Generated", val: m.count_generated },
            { label: "Temperature", val: m.temperature.toFixed(2) },
        ];

        dom.statsGrid.innerHTML = stats.map(s => `
            <div class="stat-block">
                <div class="stat-val">${s.val}</div>
                <div class="stat-lbl">${s.label}</div>
            </div>
        `).join("");
    }

    // ─── Display: Collection ──────────────────
    function displayCollection(collection) {
        if (!collection.length) {
            dom.collectionList.innerHTML =
                '<div class="text-center text-sm text-ink-500 py-6">Your haiku collection is empty.</div>';
            return;
        }

        dom.collectionList.innerHTML = collection.map((c, i) => {
            const preview = c.haikus[0] ? c.haikus[0].split("\n")[0] : "";
            const date = new Date(c.time);
            const timeStr = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

            return `
                <div class="collection-item" onclick="window.HaikuApp.loadCollectionItem(${i})">
                    <div class="flex items-center gap-2">
                        <span class="text-sm">${c.icon}</span>
                        <span class="text-sm text-ink-200 font-serif flex-1 truncate">${esc(c.theme)}</span>
                        <span class="text-[0.6rem] text-ink-500">${c.count}</span>
                    </div>
                    <div class="text-xs text-ink-500 mt-1 pl-7 italic truncate">${esc(preview)}</div>
                    <div class="text-[0.55rem] text-ink-600 mt-0.5 pl-7">${timeStr}</div>
                </div>
            `;
        }).join("");

        // Store collection data for loading
        state.collectionData = collection;
    }

    // ─── Actions ──────────────────────────────
    function copySingle(index) {
        if (!state.lastResult) return;
        const h = state.lastResult.haikus[index];
        if (!h) return;
        navigator.clipboard.writeText(h.text).then(() => {
            showToast("Haiku copied!", "success");
        }).catch(() => {
            showToast("Copy failed", "error");
        });
    }

    function copyAll() {
        if (!state.lastResult) return;
        const text = state.lastResult.haikus
            .map((h, i) => `[${i + 1}]\n${h.text}`)
            .join("\n\n");
        navigator.clipboard.writeText(text).then(() => {
            showToast("All haikus copied!", "success");
        }).catch(() => {
            showToast("Copy failed", "error");
        });
    }

    function download() {
        if (!state.lastResult) return;
        const meta = state.lastResult.meta;
        let content = `Haiku Collection — "${meta.theme}"\n`;
        content += `Mood: ${meta.mood_label} · Season: ${meta.season_label}\n`;
        content += `Generated: ${new Date(meta.timestamp).toLocaleString()}\n`;
        content += "═".repeat(40) + "\n\n";

        state.lastResult.haikus.forEach((h, i) => {
            content += `[${i + 1}]  (Score: ${h.score} pts)\n`;
            content += h.text + "\n";
            const pattern = h.syllables.join("-");
            content += `Syllables: ${pattern} ${h.is_valid_575 ? "✓ Valid 5-7-5" : "✗ Non-standard"}\n\n`;
        });

        content += "═".repeat(40) + "\n";
        content += `Model: ${meta.model}\n`;
        content += `Time: ${meta.time_seconds}s · Attempts: ${meta.attempts}\n`;

        const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `haiku_${meta.theme.replace(/\s+/g, "_")}_${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
        showToast("Haiku file downloaded!", "success");
    }

    function loadCollectionItem(index) {
        if (!state.collectionData || !state.collectionData[index]) return;
        const item = state.collectionData[index];
        dom.themeInput.value = item.theme;
        updateCharCount();
        showToast(`Loaded theme: ${item.theme}`, "success");
    }

    // ─── Initialize ───────────────────────────
    function init() {
        cacheDom();
        bindInputs();
        initParticles();
        fetchStatus();
        fetchThemes();
        fetchCollection();
    }

    // ─── Public API ───────────────────────────
    window.HaikuApp = {
        generate,
        copyAll,
        copySingle,
        download,
        loadCollectionItem,
    };

    // Boot
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
