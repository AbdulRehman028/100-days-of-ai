(() => {
    "use strict";

    const store = {
        status: null,
        ideas: [],
        promptInfo: null,
        meta: null,
        isLoading: false,
        loadingStart: 0,
        timerRef: null,
    };

    const dom = {};
    const $ = (sel) => document.querySelector(sel);

    function cacheDom() {
        dom.modelName = $("#modelName");
        dom.deviceName = $("#deviceName");
        dom.generatedCount = $("#generatedCount");

        dom.industrySelect = $("#industrySelect");
        dom.audienceInput = $("#audienceInput");
        dom.audienceCount = $("#audienceCount");
        dom.constraintsInput = $("#constraintsInput");
        dom.toneSelect = $("#toneSelect");
        dom.logoStyleSelect = $("#logoStyleSelect");
        dom.countSelect = $("#countSelect");
        dom.tempSlider = $("#tempSlider");
        dom.tempValue = $("#tempValue");

        dom.generateBtn = $("#generateBtn");

        dom.resultsTitle = $("#resultsTitle");
        dom.resultsMeta = $("#resultsMeta");
        dom.resultsGrid = $("#resultsGrid");
        dom.emptyState = $("#emptyState");

        dom.systemPrompt = $("#systemPrompt");
        dom.userPrompt = $("#userPrompt");
        dom.techniques = $("#techniques");

        dom.loadingOverlay = $("#loadingOverlay");
        dom.loadingStep = $("#loadingStep");
        dom.loadingTimer = $("#loadingTimer");
        dom.toast = $("#toast");
    }

    function escapeHtml(v) {
        const d = document.createElement("div");
        d.textContent = v ?? "";
        return d.innerHTML;
    }

    function showToast(msg, type = "success") {
        dom.toast.className = `toast ${type} show`;
        dom.toast.textContent = msg;
        setTimeout(() => dom.toast.classList.remove("show"), 2500);
    }

    function setLoading(active) {
        store.isLoading = active;
        dom.generateBtn.disabled = active;

        if (active) {
            store.loadingStart = performance.now();
            dom.loadingOverlay.classList.add("active");
            dom.loadingStep.textContent = "Generating startup names and logos...";
            store.timerRef = setInterval(() => {
                const sec = ((performance.now() - store.loadingStart) / 1000).toFixed(1);
                dom.loadingTimer.textContent = `${sec}s`;
                if (sec > 8) dom.loadingStep.textContent = "Refining ideas and rendering logos...";
            }, 100);
        } else {
            dom.loadingOverlay.classList.remove("active");
            clearInterval(store.timerRef);
            store.timerRef = null;
        }
    }

    function fillSelect(el, values, fallback) {
        el.innerHTML = "";
        values.forEach((v) => {
            const o = document.createElement("option");
            o.value = v;
            o.textContent = v;
            if (v === fallback) o.selected = true;
            el.appendChild(o);
        });
    }

    function bindEvents() {
        dom.audienceInput.addEventListener("input", () => {
            dom.audienceCount.textContent = `${dom.audienceInput.value.length}/80`;
        });

        dom.tempSlider.addEventListener("input", () => {
            dom.tempValue.textContent = Number(dom.tempSlider.value).toFixed(2);
        });

        dom.audienceInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                generateIdeas();
            }
        });
    }

    async function fetchStatus() {
        const res = await fetch("/status");
        if (!res.ok) throw new Error("Failed to load app status");
        const data = await res.json();
        store.status = data;

        dom.modelName.textContent = data.model || "Unknown";
        dom.deviceName.textContent = data.device || "CPU";
        dom.generatedCount.textContent = data.generated_total || 0;

        fillSelect(dom.industrySelect, data.industries || [], "ai tooling");
        fillSelect(dom.toneSelect, data.tones || [], "minimal");
        fillSelect(dom.logoStyleSelect, data.logo_styles || [], "monogram");

        if (!dom.audienceInput.value) dom.audienceInput.value = "early-stage teams";
        dom.audienceCount.textContent = `${dom.audienceInput.value.length}/80`;
    }

    function buildCard(idea, idx) {
        const card = document.createElement("article");
        card.className = "idea-card";
        card.style.animationDelay = `${idx * 60}ms`;

        card.innerHTML = `
            <div class="idea-top">
                <p class="idea-name">${escapeHtml(idea.name)}</p>
                <p class="idea-tagline">${escapeHtml(idea.tagline)}</p>
            </div>
            <div class="idea-body">
                <div class="logo-box">
                    <img src="${idea.logo_data_uri}" alt="${escapeHtml(idea.name)} logo" />
                </div>
                <div class="idea-row">
                    <p>Problem</p>
                    <p>${escapeHtml(idea.problem)}</p>
                </div>
                <div class="idea-row">
                    <p>Solution</p>
                    <p>${escapeHtml(idea.solution)}</p>
                </div>
                <div class="idea-row">
                    <p>Logo Concept</p>
                    <p>${escapeHtml(idea.logo_concept)}</p>
                </div>
                <div class="card-actions">
                    <button class="card-btn" data-action="copy" data-index="${idx}">Copy Idea</button>
                    <button class="card-btn" data-action="download-logo" data-index="${idx}">Download Logo</button>
                </div>
            </div>
        `;

        card.querySelector('[data-action="copy"]').addEventListener("click", () => copyIdea(idx));
        card.querySelector('[data-action="download-logo"]').addEventListener("click", () => downloadLogo(idx));

        return card;
    }

    function renderIdeas() {
        dom.resultsGrid.innerHTML = "";

        if (!store.ideas.length) {
            dom.emptyState.classList.remove("hidden");
            dom.resultsTitle.textContent = "No ideas yet";
            dom.resultsMeta.textContent = "Fill inputs and generate your startup batch.";
            return;
        }

        dom.emptyState.classList.add("hidden");
        dom.resultsTitle.textContent = `${store.ideas.length} startup concepts generated`;
        dom.resultsMeta.textContent = `${store.meta.industry} · ${store.meta.tone} tone · ${store.meta.time_seconds}s`;

        store.ideas.forEach((idea, idx) => {
            dom.resultsGrid.appendChild(buildCard(idea, idx));
        });

        dom.generatedCount.textContent = store.meta.generated_total;
    }

    function renderPromptInfo() {
        if (!store.promptInfo) return;

        dom.systemPrompt.textContent = store.promptInfo.system_prompt || "";
        dom.userPrompt.textContent = store.promptInfo.user_prompt || "";

        dom.techniques.innerHTML = "";
        (store.promptInfo.techniques || []).forEach((t) => {
            const span = document.createElement("span");
            span.className = "tech-tag";
            span.textContent = t;
            dom.techniques.appendChild(span);
        });
    }

    function currentPayload() {
        return {
            industry: dom.industrySelect.value,
            audience: dom.audienceInput.value.trim(),
            constraints: dom.constraintsInput.value.trim(),
            tone: dom.toneSelect.value,
            logo_style: dom.logoStyleSelect.value,
            count: parseInt(dom.countSelect.value, 10),
            temperature: parseFloat(dom.tempSlider.value),
        };
    }

    async function generateIdeas() {
        if (store.isLoading) return;

        const payload = currentPayload();
        if (!payload.audience) {
            showToast("Please enter target audience", "error");
            dom.audienceInput.focus();
            return;
        }

        setLoading(true);

        try {
            const res = await fetch("/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Generation failed");

            store.ideas = data.ideas || [];
            store.meta = data.meta || {};
            store.promptInfo = data.prompt_info || null;

            renderIdeas();
            renderPromptInfo();
            showToast(`Generated ${store.ideas.length} startup ideas`, "success");
        } catch (err) {
            showToast(err.message || "Something went wrong", "error");
        } finally {
            setLoading(false);
        }
    }

    function toIdeaText(idea, idx) {
        return [
            `IDEA ${idx + 1}`,
            `Name: ${idea.name}`,
            `Tagline: ${idea.tagline}`,
            `Problem: ${idea.problem}`,
            `Solution: ${idea.solution}`,
            `Logo Concept: ${idea.logo_concept}`,
        ].join("\n");
    }

    function copyIdea(index) {
        const idea = store.ideas[index];
        if (!idea) return;
        navigator.clipboard
            .writeText(toIdeaText(idea, index))
            .then(() => showToast("Idea copied", "success"))
            .catch(() => showToast("Copy failed", "error"));
    }

    function copyBatch() {
        if (!store.ideas.length) {
            showToast("No ideas to copy", "error");
            return;
        }
        const block = store.ideas.map((idea, idx) => toIdeaText(idea, idx)).join("\n\n---\n\n");
        navigator.clipboard
            .writeText(block)
            .then(() => showToast("Batch copied", "success"))
            .catch(() => showToast("Copy failed", "error"));
    }

    function downloadLogo(index) {
        const idea = store.ideas[index];
        if (!idea) return;

        const blob = new Blob([idea.logo_svg], { type: "image/svg+xml;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${idea.name.toLowerCase().replace(/[^a-z0-9]+/g, "-") || "logo"}.svg`;
        a.click();
        URL.revokeObjectURL(url);
        showToast("Logo downloaded", "success");
    }

    function downloadBatch() {
        if (!store.ideas.length) {
            showToast("No ideas to download", "error");
            return;
        }

        const payload = {
            meta: store.meta,
            ideas: store.ideas,
            exported_at: new Date().toISOString(),
        };
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `startup_batch_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast("Batch JSON downloaded", "success");
    }

    async function init() {
        cacheDom();
        bindEvents();
        try {
            await fetchStatus();
        } catch (err) {
            showToast("Failed to initialize app", "error");
        }
    }

    window.StartupApp = {
        generateIdeas,
        copyBatch,
        downloadBatch,
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
