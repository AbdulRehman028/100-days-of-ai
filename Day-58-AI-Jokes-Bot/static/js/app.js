/* ══════════════════════════════════════════════
   AI Jokes Bot — Client Logic
   Day 58 · Prompt Engineering
   ══════════════════════════════════════════════ */

// ── State ─────────────────────────────────────
let selectedStyle = "one-liner";
let lastResult = null;
let loadingTimer = null;

// ── Init ──────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    fetchStatus();
    fetchTopics();
    setupListeners();
});

// ── Setup Listeners ───────────────────────────
function setupListeners() {
    const topicInput = document.getElementById("topicInput");
    const charCount = document.getElementById("charCount");
    const tempSlider = document.getElementById("tempSlider");
    const tempValue = document.getElementById("tempValue");

    topicInput.addEventListener("input", () => {
        charCount.textContent = `${topicInput.value.length} / 100`;
    });

    topicInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            generateJokes();
        }
    });

    tempSlider.addEventListener("input", () => {
        tempValue.textContent = parseFloat(tempSlider.value).toFixed(2);
    });
}

// ── Fetch Status ──────────────────────────────
async function fetchStatus() {
    try {
        const res = await fetch("/status");
        const data = await res.json();

        document.getElementById("engineModel").textContent = data.model || "—";
        document.getElementById("engineDevice").textContent = data.device || "—";
        document.getElementById("totalJokes").textContent = data.jokes_generated || 0;

        // Build style grid
        if (data.styles) {
            buildStyleGrid(data.styles);
        }

        // Fetch history
        fetchHistory();
    } catch (e) {
        document.getElementById("engineModel").textContent = "Connection error";
    }
}

// ── Build Style Grid ──────────────────────────
function buildStyleGrid(styles) {
    const grid = document.getElementById("styleGrid");
    grid.innerHTML = "";

    for (const [key, style] of Object.entries(styles)) {
        const chip = document.createElement("div");
        chip.className = `style-chip${key === selectedStyle ? " active" : ""}`;
        chip.dataset.style = key;
        chip.innerHTML = `
            <div class="style-icon">${style.icon}</div>
            <div class="style-name">${style.label}</div>
            <div class="style-desc">${style.desc}</div>
        `;
        chip.onclick = () => selectStyle(key);
        grid.appendChild(chip);
    }
}

function selectStyle(style) {
    selectedStyle = style;
    document.querySelectorAll(".style-chip").forEach(c => {
        c.classList.toggle("active", c.dataset.style === style);
    });
}

// ── Fetch Topics ──────────────────────────────
async function fetchTopics() {
    try {
        const res = await fetch("/topics");
        const data = await res.json();
        const container = document.getElementById("topicChips");
        container.innerHTML = "";

        // Show random 10
        const shuffled = data.topics.sort(() => 0.5 - Math.random()).slice(0, 12);
        shuffled.forEach(topic => {
            const chip = document.createElement("span");
            chip.className = "topic-chip";
            chip.textContent = topic;
            chip.onclick = () => {
                document.getElementById("topicInput").value = topic;
                document.getElementById("charCount").textContent = `${topic.length} / 100`;
            };
            container.appendChild(chip);
        });
    } catch (e) {
        console.error("Failed to fetch topics:", e);
    }
}

// ── Generate Jokes ────────────────────────────
async function generateJokes() {
    const topic = document.getElementById("topicInput").value.trim();
    if (!topic) {
        showToast("Please enter a topic!", "error");
        document.getElementById("topicInput").focus();
        return;
    }

    const count = parseInt(document.getElementById("countSelect").value);
    const temperature = parseFloat(document.getElementById("tempSlider").value);

    // Show loading
    showLoading(true);
    document.getElementById("generateBtn").disabled = true;

    const loadingSteps = [
        "🔧 Engineering the prompt...",
        "📝 Applying few-shot examples...",
        "🎤 Generating comedy gold...",
        "✨ Polishing punchlines...",
    ];
    let stepIdx = 0;
    const stepInterval = setInterval(() => {
        stepIdx = (stepIdx + 1) % loadingSteps.length;
        document.getElementById("loadingStep").textContent = loadingSteps[stepIdx];
    }, 2500);

    try {
        const res = await fetch("/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ topic, style: selectedStyle, count, temperature }),
        });

        const data = await res.json();
        clearInterval(stepInterval);

        if (data.error) {
            showToast(data.error, "error");
        } else {
            lastResult = data;
            displayJokes(data);
            displayPromptEngineering(data.prompt_engineering);
            displayStats(data.meta);
            showToast(`${data.jokes.length} joke${data.jokes.length !== 1 ? 's' : ''} generated!`, "success");
            fetchHistory();
            fetchStatus(); // Update total count
        }
    } catch (e) {
        clearInterval(stepInterval);
        showToast("Generation failed — check the server.", "error");
    } finally {
        showLoading(false);
        document.getElementById("generateBtn").disabled = false;
    }
}

// ── Display Jokes ─────────────────────────────
function displayJokes(data) {
    const emptyState = document.getElementById("emptyState");
    const display = document.getElementById("jokesDisplay");

    emptyState.style.display = "none";
    display.style.display = "block";

    // Header
    document.getElementById("jokesTitle").textContent =
        `${data.meta.style_icon} ${data.meta.style_label} Jokes: "${data.meta.topic}"`;
    document.getElementById("jokesMeta").textContent =
        `${data.meta.count_generated} joke${data.meta.count_generated !== 1 ? 's' : ''} · ${data.meta.time_seconds}s · ${data.meta.model.split('/').pop()}`;

    // Jokes list
    const list = document.getElementById("jokesList");
    list.innerHTML = "";

    data.jokes.forEach((joke, i) => {
        const item = document.createElement("div");
        item.className = "joke-item";
        item.style.animationDelay = `${i * 0.1}s`;
        item.innerHTML = `
            <div class="joke-number">${i + 1}</div>
            <div class="joke-text">${escapeHtml(joke)}</div>
            <button class="joke-copy-btn" onclick="copySingleJoke(${i})">Copy</button>
        `;
        list.appendChild(item);
    });
}

// ── Display Prompt Engineering ─────────────────
function displayPromptEngineering(pe) {
    if (!pe) return;

    const card = document.getElementById("promptCard");
    card.hidden = false;

    document.getElementById("promptSystem").textContent = pe.system_prompt || "";
    document.getElementById("promptUser").textContent = pe.user_prompt || "";

    const techList = document.getElementById("techniquesList");
    techList.innerHTML = "";
    (pe.technique_used || []).forEach(tech => {
        const tag = document.createElement("span");
        tag.className = "technique-tag";
        tag.textContent = tech;
        techList.appendChild(tag);
    });
}

// ── Display Stats ─────────────────────────────
function displayStats(meta) {
    if (!meta) return;

    const card = document.getElementById("statsCard");
    card.hidden = false;

    const grid = document.getElementById("statsGrid");
    grid.innerHTML = `
        <div class="stat-item">
            <div class="stat-value">${meta.count_generated}</div>
            <div class="stat-label">Jokes</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${meta.time_seconds}s</div>
            <div class="stat-label">Time</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${meta.temperature}</div>
            <div class="stat-label">Temp</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${meta.style_icon}</div>
            <div class="stat-label">${meta.style_label}</div>
        </div>
    `;
}

// ── Fetch History ─────────────────────────────
async function fetchHistory() {
    try {
        const res = await fetch("/history");
        const data = await res.json();
        const list = document.getElementById("historyList");

        if (!data.history || data.history.length === 0) {
            list.innerHTML = '<div class="history-empty">No jokes yet — generate your first batch!</div>';
            return;
        }

        list.innerHTML = "";
        data.history.forEach(item => {
            const el = document.createElement("div");
            el.className = "history-item";
            const timeStr = new Date(item.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            el.innerHTML = `
                <div class="history-topic">${item.icon} ${item.topic}</div>
                <div class="history-meta">${item.style} · ${item.count} jokes · ${timeStr}</div>
            `;
            el.onclick = () => {
                document.getElementById("topicInput").value = item.topic;
                document.getElementById("charCount").textContent = `${item.topic.length} / 100`;
            };
            list.appendChild(el);
        });
    } catch (e) {
        console.error("History fetch failed:", e);
    }
}

// ── Copy / Download ───────────────────────────
function copySingleJoke(idx) {
    if (!lastResult || !lastResult.jokes[idx]) return;
    navigator.clipboard.writeText(lastResult.jokes[idx]).then(() => {
        showToast("Joke copied!", "success");
    });
}

function copyJokes() {
    if (!lastResult) return;
    const text = lastResult.jokes.map((j, i) => `${i + 1}. ${j}`).join("\n\n");
    navigator.clipboard.writeText(text).then(() => {
        showToast("All jokes copied!", "success");
    });
}

function downloadJokes() {
    if (!lastResult) return;
    const text = [
        `AI Jokes Bot — ${lastResult.meta.style_label}`,
        `Topic: ${lastResult.meta.topic}`,
        `Generated: ${new Date(lastResult.meta.timestamp).toLocaleString()}`,
        `Model: ${lastResult.meta.model}`,
        `Temperature: ${lastResult.meta.temperature}`,
        "",
        "═".repeat(40),
        "",
        ...lastResult.jokes.map((j, i) => `${i + 1}. ${j}\n`),
    ].join("\n");

    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `jokes_${lastResult.meta.topic.replace(/\s+/g, '_')}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Downloaded!", "success");
}

function regenerate() {
    generateJokes();
}

// ── Loading / Toast ───────────────────────────
function showLoading(show) {
    const overlay = document.getElementById("loadingOverlay");
    const timerEl = document.getElementById("loadingTimer");

    if (show) {
        overlay.classList.add("active");
        let t = 0;
        loadingTimer = setInterval(() => {
            t += 0.1;
            timerEl.textContent = t.toFixed(1) + "s";
        }, 100);
    } else {
        overlay.classList.remove("active");
        if (loadingTimer) {
            clearInterval(loadingTimer);
            loadingTimer = null;
        }
    }
}

function showToast(msg, type = "") {
    const toast = document.getElementById("toast");
    toast.textContent = msg;
    toast.className = `toast show ${type}`;
    setTimeout(() => toast.className = "toast", 3000);
}

// ── Utility ───────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
