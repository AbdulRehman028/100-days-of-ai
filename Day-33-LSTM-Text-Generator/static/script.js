// AI Text Generator JavaScript

let currentType = 'story';
let generatedText = '';

// Example prompts for each content type
const examplePrompts = {
    story: [
        { text: 'A brave knight embarks on a quest to save the kingdom', icon: 'fa-chess-knight', label: "Knight's Quest" },
        { text: 'A time traveler accidentally changes history', icon: 'fa-clock', label: 'Time Travel' },
        { text: 'A detective solving a mystery in a futuristic city', icon: 'fa-search', label: 'Cyber Detective' },
        { text: 'Two rival chefs competing in a magical cooking contest', icon: 'fa-utensils', label: 'Culinary Magic' }
    ],
    poem: [
        { text: 'The moon whispers secrets to the stars', icon: 'fa-moon', label: 'Moon Poem' },
        { text: 'Ocean waves dancing with the sunset', icon: 'fa-water', label: 'Ocean Sunset' },
        { text: 'A butterfly\'s journey through seasons', icon: 'fa-leaf', label: 'Butterfly Life' },
        { text: 'Silent snowfall on a winter night', icon: 'fa-snowflake', label: 'Winter Night' }
    ],
    script: [
        { text: 'INT. SPACESHIP - A captain discovers an alien artifact', icon: 'fa-space-shuttle', label: 'Sci-Fi Script' },
        { text: 'EXT. COFFEE SHOP - Two old friends reunite after 10 years', icon: 'fa-coffee', label: 'Reunion Scene' },
        { text: 'INT. LABORATORY - A scientist makes a breakthrough discovery', icon: 'fa-flask', label: 'Lab Discovery' },
        { text: 'EXT. MOUNTAIN TOP - A hero faces their final challenge', icon: 'fa-mountain', label: 'Final Challenge' }
    ],
    quest: [
        { text: 'Find the ancient treasure hidden in the enchanted forest', icon: 'fa-gem', label: 'Treasure Hunt' },
        { text: 'Defeat the dragon terrorizing the village', icon: 'fa-dragon', label: 'Dragon Slayer' },
        { text: 'Rescue the prince from the ice palace', icon: 'fa-snowflake', label: 'Ice Rescue' },
        { text: 'Collect three magical artifacts to save the realm', icon: 'fa-magic', label: 'Artifact Quest' }
    ],
    social: [
        { text: '5 life-changing habits that will transform your productivity', icon: 'fa-fire', label: 'Productivity Tips' },
        { text: 'Why morning routines are overrated - my honest take', icon: 'fa-sun', label: 'Morning Routine' },
        { text: 'Just launched my first product! Here\'s what I learned', icon: 'fa-rocket', label: 'Product Launch' },
        { text: 'The one skill that changed my career completely', icon: 'fa-chart-line', label: 'Career Growth' }
    ],
    blog: [
        { text: 'The complete guide to starting your first online business in 2025', icon: 'fa-laptop-code', label: 'Online Business' },
        { text: 'How I built a side hustle that makes $5K per month', icon: 'fa-dollar-sign', label: 'Side Hustle' },
        { text: '10 essential tools every content creator needs', icon: 'fa-tools', label: 'Creator Tools' },
        { text: 'From beginner to expert: My coding journey in 12 months', icon: 'fa-code', label: 'Coding Journey' }
    ],
    email: [
        { text: 'Meeting request to discuss the Q1 marketing strategy', icon: 'fa-briefcase', label: 'Meeting Request' },
        { text: 'Project update and next steps for the team', icon: 'fa-tasks', label: 'Project Update' },
        { text: 'Thank you email to a client after successful project completion', icon: 'fa-handshake', label: 'Thank You' },
        { text: 'Follow-up on job application for Software Developer position', icon: 'fa-file-alt', label: 'Job Follow-up' }
    ],
    article: [
        { text: 'How artificial intelligence is revolutionizing healthcare', icon: 'fa-heartbeat', label: 'AI Healthcare' },
        { text: 'The future of renewable energy and sustainability', icon: 'fa-solar-panel', label: 'Green Energy' },
        { text: 'Understanding cryptocurrency and blockchain technology', icon: 'fa-bitcoin', label: 'Crypto Explained' },
        { text: 'The impact of remote work on modern workplace culture', icon: 'fa-home', label: 'Remote Work' }
    ]
};

document.addEventListener('DOMContentLoaded', function() {
    // Setup form submission
    const form = document.getElementById('generatorForm');
    form.addEventListener('submit', handleSubmit);
    
    // Setup type buttons
    const typeButtons = document.querySelectorAll('.type-btn');
    typeButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            typeButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            currentType = this.dataset.type;
            updateExamples(currentType);
        });
    });
    
    // Initialize examples for default type
    updateExamples(currentType);
    
    // Setup character counter
    const textarea = document.getElementById('promptInput');
    textarea.addEventListener('input', updateCharCount);
    
    // Setup advanced settings toggle
    const toggleBtn = document.getElementById('toggleAdvanced');
    const settingsPanel = document.getElementById('settingsPanel');
    toggleBtn.addEventListener('click', function() {
        this.classList.toggle('active');
        settingsPanel.classList.toggle('active');
    });
    
    // Setup temperature slider
    const tempSlider = document.getElementById('temperature');
    const tempValue = document.getElementById('tempValue');
    tempSlider.addEventListener('input', function() {
        tempValue.textContent = this.value;
    });
    
    // Setup length slider
    const lengthSlider = document.getElementById('maxLength');
    const lengthValue = document.getElementById('lengthValue');
    lengthSlider.addEventListener('input', function() {
        lengthValue.textContent = this.value;
    });
    
    // Load model stats
    loadStats();
});

// Update examples based on selected type
function updateExamples(type) {
    const examplesGrid = document.querySelector('.examples-grid');
    const examples = examplePrompts[type] || examplePrompts.story;
    
    examplesGrid.innerHTML = examples.map((example, index) => `
        <button class="example-btn" onclick="tryExample('${example.text.replace(/'/g, "\\'")}', '${type}')" style="animation-delay: ${index * 0.1}s">
            <i class="fas ${example.icon}"></i>
            <span>${example.label}</span>
        </button>
    `).join('');
}

function updateCharCount() {
    const textarea = document.getElementById('promptInput');
    const charCount = document.getElementById('charCount');
    charCount.textContent = textarea.value.length;
}

async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        document.getElementById('modelStatus').textContent = data.model.toUpperCase() + ' Ready';
    } catch (error) {
        console.error('Error loading stats:', error);
        document.getElementById('modelStatus').textContent = 'Ready';
    }
}

async function handleSubmit(e) {
    e.preventDefault();
    
    const promptInput = document.getElementById('promptInput');
    const prompt = promptInput.value.trim();
    
    if (!prompt) {
        showError('Please enter a prompt to generate text');
        return;
    }
    
    if (prompt.length < 3) {
        showError('Prompt is too short. Please enter at least 3 characters.');
        return;
    }
    
    const temperature = parseFloat(document.getElementById('temperature').value);
    const maxLength = parseInt(document.getElementById('maxLength').value);
    
    const generateBtn = document.getElementById('generateBtn');
    const originalContent = generateBtn.innerHTML;
    generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Generating...</span>';
    generateBtn.classList.add('loading');
    generateBtn.disabled = true;
    
    document.getElementById('resultSection').style.display = 'none';
    
    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                type: currentType,
                temperature: temperature,
                max_length: maxLength
            }),
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResult(data);
            // Scroll to result
            setTimeout(() => {
                document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        } else {
            showError(data.error || 'Failed to generate text');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Network error. Please try again.');
    } finally {
        generateBtn.innerHTML = originalContent;
        generateBtn.classList.remove('loading');
        generateBtn.disabled = false;
    }
}

function displayResult(data) {
    generatedText = data.generated_text;
    
    // Set result content
    document.getElementById('resultPrompt').textContent = data.prompt;
    document.getElementById('resultContent').textContent = data.generated_text;
    
    // Set type badge
    const typeBadge = document.getElementById('resultType');
    typeBadge.textContent = data.type.charAt(0).toUpperCase() + data.type.slice(1);
    
    // Set generation time
    document.getElementById('resultTime').innerHTML = `<i class="fas fa-clock"></i> ${data.generation_time}s`;
    
    // Show result section
    const resultSection = document.getElementById('resultSection');
    resultSection.style.display = 'block';
}

// Show error message
function showError(message) {
    alert('Error: ' + message);
}

// Copy to clipboard
function copyToClipboard() {
    if (!generatedText) return;
    
    // Try modern clipboard API first
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(generatedText).then(() => {
            showCopySuccess();
        }).catch(err => {
            console.error('Clipboard API failed:', err);
            fallbackCopy();
        });
    } else {
        // Use fallback method for non-secure contexts
        fallbackCopy();
    }
}

// Fallback copy method using textarea
function fallbackCopy() {
    try {
        // Create temporary textarea
        const textarea = document.createElement('textarea');
        textarea.value = generatedText;
        textarea.style.position = 'fixed';
        textarea.style.top = '-9999px';
        textarea.style.left = '-9999px';
        document.body.appendChild(textarea);
        
        // Select and copy
        textarea.focus();
        textarea.select();
        
        const successful = document.execCommand('copy');
        document.body.removeChild(textarea);
        
        if (successful) {
            showCopySuccess();
        } else {
            throw new Error('execCommand failed');
        }
    } catch (err) {
        console.error('Fallback copy failed:', err);
        // Show manual copy option
        prompt('Copy this text manually:', generatedText);
    }
}

// Show copy success feedback
function showCopySuccess() {
    const btn = event.target.closest('.action-btn');
    const originalContent = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
    btn.style.background = '#10b981';
    btn.style.borderColor = '#10b981';
    
    setTimeout(() => {
        btn.innerHTML = originalContent;
        btn.style.background = '';
        btn.style.borderColor = '';
    }, 2000);
}

// Download text as file
function downloadText() {
    if (!generatedText) return;
    
    const element = document.createElement('a');
    const file = new Blob([generatedText], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = `generated_${currentType}_${Date.now()}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
}

// Generate again with same settings
function generateAgain() {
    const promptInput = document.getElementById('promptInput');
    if (promptInput.value.trim()) {
        handleSubmit(new Event('submit'));
    }
}

// Try example prompt
function tryExample(prompt, type) {
    // Set prompt
    document.getElementById('promptInput').value = prompt;
    updateCharCount();
    
    // Set type
    const typeButtons = document.querySelectorAll('.type-btn');
    typeButtons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.type === type) {
            btn.classList.add('active');
            currentType = type;
        }
    });
    
    // Scroll to form
    document.querySelector('.generator-section').scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    // Focus textarea
    document.getElementById('promptInput').focus();
}