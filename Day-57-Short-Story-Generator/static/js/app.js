// ============================================
// StoryForge AI — Client-Side Application
// ============================================

// ── State ──
let selectedGenre = 'fantasy';
let currentStory = null;
let samplePrompts = [];

// ============================================
// INIT
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    loadGenres();
    loadStatus();
    loadPrompts();
    loadHistory();
    setupListeners();
});

function setupListeners() {
    const plotInput = document.getElementById('plotInput');
    plotInput.addEventListener('input', () => {
        const len = plotInput.value.length;
        document.getElementById('charCount').textContent = `${len} / 500`;
    });

    document.getElementById('creativitySlider').addEventListener('input', (e) => {
        document.getElementById('creativityValue').textContent = e.target.value;
    });
}

// ============================================
// LOAD DATA
// ============================================
async function loadGenres() {
    try {
        const res = await fetch('/status');
        const data = await res.json();
        const genres = data.genres;
        const grid = document.getElementById('genreGrid');
        grid.innerHTML = '';

        for (const [key, genre] of Object.entries(genres)) {
            const chip = document.createElement('div');
            chip.className = `genre-chip ${key === selectedGenre ? 'active' : ''}`;
            chip.dataset.genre = key;
            chip.innerHTML = `<span class="g-icon">${genre.icon}</span><span class="g-label">${genre.label}</span>`;
            chip.onclick = () => selectGenre(key);
            grid.appendChild(chip);
        }
    } catch (e) {
        console.error('Failed to load genres:', e);
    }
}

function selectGenre(genre) {
    selectedGenre = genre;
    document.querySelectorAll('.genre-chip').forEach(c => {
        c.classList.toggle('active', c.dataset.genre === genre);
    });
}

async function loadStatus() {
    try {
        const res = await fetch('/status');
        const data = await res.json();
        document.getElementById('engineInfo').textContent = data.engine.framework;
        document.getElementById('deviceInfo').textContent = data.engine.device;
        document.getElementById('totalStories').textContent = data.engine.stories_generated;
    } catch (e) {}
}

async function loadPrompts() {
    try {
        const res = await fetch('/prompts');
        const data = await res.json();
        samplePrompts = data.prompts;
        renderInspirationChips();
    } catch (e) {}
}

function renderInspirationChips() {
    const container = document.getElementById('inspirationChips');
    const shuffled = [...samplePrompts].sort(() => 0.5 - Math.random()).slice(0, 5);
    container.innerHTML = shuffled.map(p =>
        `<div class="inspiration-chip" onclick="usePrompt('${p.plot.replace(/'/g, "\\'")}', '${p.genre}')">${p.plot.substring(0, 50)}...</div>`
    ).join('');
}

function usePrompt(plot, genre) {
    document.getElementById('plotInput').value = plot;
    selectGenre(genre);
    document.getElementById('charCount').textContent = `${plot.length} / 500`;
}

async function loadHistory() {
    try {
        const res = await fetch('/history');
        const data = await res.json();
        const container = document.getElementById('historyList');

        if (data.stories.length === 0) return;

        container.innerHTML = data.stories.map(s => `
            <div class="history-item" onclick="showHistoryStory(this)" data-story='${JSON.stringify(s).replace(/'/g, "&apos;")}'>
                <div>
                    <div class="history-title">${s.title || 'Untitled'}</div>
                    <div class="history-meta">${s.genre || ''} · ${s.word_count || 0} words</div>
                </div>
            </div>
        `).join('');
    } catch (e) {}
}

function showHistoryStory(el) {
    try {
        const story = JSON.parse(el.dataset.story.replace(/&apos;/g, "'"));
        displayStory({
            success: true,
            title: story.title,
            story: story.story,
            genre: story.genre,
            genre_icon: '',
            tone: story.tone,
            word_count: story.word_count,
            generation_time: story.generation_time,
            outline: '',
            chain_steps: [],
            analysis: null,
        });
    } catch (e) {
        console.error('Error showing history story:', e);
    }
}

// ============================================
// GENERATE
// ============================================
async function generateStory() {
    const plot = document.getElementById('plotInput').value.trim();
    if (!plot) {
        showToast('Please enter a plot premise!', 'error');
        return;
    }

    const btn = document.getElementById('generateBtn');
    btn.disabled = true;
    btn.querySelector('.btn-text').textContent = '⏳ Generating...';

    // Show loading overlay
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.add('show');

    // Animate chain steps
    const steps = ['📋 Planning Outline...', '📝 Writing Story...', '✨ Generating Title...', '✅ Polishing Output...'];
    let stepIdx = 0;
    let elapsed = 0;

    const loadTimer = setInterval(() => {
        elapsed += 0.1;
        document.getElementById('loadingTimer').textContent = `${elapsed.toFixed(1)}s`;
    }, 100);

    const stepTimer = setInterval(() => {
        if (stepIdx < 4) {
            document.getElementById('loadingStep').textContent = steps[Math.min(stepIdx, 3)];

            // Update dots
            for (let i = 0; i < 4; i++) {
                const dot = document.getElementById(`dot${i + 1}`);
                if (i < stepIdx) dot.className = 'load-dot done';
                else if (i === stepIdx) dot.className = 'load-dot active';
                else dot.className = 'load-dot';
            }

            // Update chain pipeline steps
            document.querySelectorAll('.chain-step').forEach((el, i) => {
                if (i < stepIdx) el.className = 'chain-step done';
                else if (i === stepIdx) el.className = 'chain-step active';
                else el.className = 'chain-step';
            });

            stepIdx++;
        }
    }, 3000);

    try {
        const res = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                plot,
                genre: selectedGenre,
                tone: document.getElementById('toneSelect').value,
                length: document.getElementById('lengthSelect').value,
                creativity: parseFloat(document.getElementById('creativitySlider').value),
            })
        });

        const data = await res.json();

        clearInterval(loadTimer);
        clearInterval(stepTimer);
        overlay.classList.remove('show');

        // Set all chain steps to done
        document.querySelectorAll('.chain-step').forEach(el => el.className = 'chain-step done');

        if (data.success) {
            displayStory(data);
            showToast(`Story generated in ${data.generation_time}s!`, 'success');
            loadHistory();
            loadStatus();
        } else {
            showToast(data.error || 'Generation failed', 'error');
        }
    } catch (e) {
        clearInterval(loadTimer);
        clearInterval(stepTimer);
        overlay.classList.remove('show');
        showToast('Connection error', 'error');
    }

    btn.disabled = false;
    btn.querySelector('.btn-text').textContent = '✨ Generate Story';
}

// ============================================
// DISPLAY
// ============================================
function displayStory(data) {
    currentStory = data;

    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('storyDisplay').style.display = 'block';

    // Title
    document.getElementById('storyTitle').textContent = data.title;

    // Meta badges
    const meta = document.getElementById('storyMeta');
    meta.innerHTML = `
        <span class="meta-badge">${data.genre_icon || '📖'} ${data.genre}</span>
        ${data.tone ? `<span class="meta-badge">🎭 ${data.tone}</span>` : ''}
        <span class="meta-badge">📝 ${data.word_count} words</span>
        ${data.generation_time ? `<span class="meta-badge">⚡ ${data.generation_time}s</span>` : ''}
    `;

    // Outline
    if (data.outline) {
        document.getElementById('outlineText').textContent = data.outline;
    }

    // Story body
    document.getElementById('storyBody').textContent = data.story;

    // Stats
    if (data.analysis) {
        const stats = document.getElementById('statsCard');
        stats.hidden = false;
        const grid = document.getElementById('statsGrid');
        const a = data.analysis;
        grid.innerHTML = `
            <div class="glass-card stat-card">
                <div class="stat-value">${a.word_count}</div>
                <div class="stat-label">Words</div>
            </div>
            <div class="glass-card stat-card">
                <div class="stat-value">${a.sentence_count}</div>
                <div class="stat-label">Sentences</div>
            </div>
            <div class="glass-card stat-card">
                <div class="stat-value">${a.paragraph_count}</div>
                <div class="stat-label">Paragraphs</div>
            </div>
            <div class="glass-card stat-card">
                <div class="stat-value">${a.vocab_richness}%</div>
                <div class="stat-label">Vocabulary</div>
            </div>
            <div class="glass-card stat-card">
                <div class="stat-value">${a.avg_sentence_length}</div>
                <div class="stat-label">Avg Sent. Len</div>
            </div>
            <div class="glass-card stat-card">
                <div class="stat-value">${a.dialogue_lines}</div>
                <div class="stat-label">Dialogue</div>
            </div>
            <div class="glass-card stat-card">
                <div class="stat-value">${a.descriptive_words}</div>
                <div class="stat-label">Descriptive</div>
            </div>
            <div class="glass-card stat-card">
                <div class="stat-value">${a.unique_words}</div>
                <div class="stat-label">Unique Words</div>
            </div>
        `;
    }

    // Scroll to story
    document.getElementById('storyCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function toggleOutline() {
    const content = document.getElementById('outlineContent');
    const arrow = document.getElementById('outlineArrow');
    content.classList.toggle('open');
    arrow.textContent = content.classList.contains('open') ? '▾' : '▸';
}

// ============================================
// ACTIONS
// ============================================
function copyStory() {
    if (!currentStory) return;
    const text = `${currentStory.title}\n\n${currentStory.story}`;
    navigator.clipboard.writeText(text).then(() => {
        showToast('Story copied to clipboard!', 'success');
    });
}

function downloadStory() {
    if (!currentStory) return;
    const text = `${currentStory.title}\n${'='.repeat(currentStory.title.length)}\n\nGenre: ${currentStory.genre}\n\n---\n\n${currentStory.story}`;
    const blob = new Blob([text], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${currentStory.title.replace(/[^a-zA-Z0-9]/g, '_')}.txt`;
    a.click();
}

function showToast(msg, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.className = `toast ${type} show`;
    setTimeout(() => toast.classList.remove('show'), 3000);
}
