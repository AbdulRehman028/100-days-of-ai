// Emoji Prediction JavaScript

let currentText = '';
let predictedEmojis = [];

document.addEventListener('DOMContentLoaded', function() {
    // Setup form submission
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', handleSubmit);
    
    // Setup character counter
    const textarea = document.getElementById('textInput');
    textarea.addEventListener('input', updateCharCount);
    
    // Load model stats
    loadStats();
});

// Update character counter
function updateCharCount() {
    const textarea = document.getElementById('textInput');
    const charCount = document.getElementById('charCount');
    charCount.textContent = textarea.value.length;
    
    // Change color when approaching limit
    if (textarea.value.length > 450) {
        charCount.style.color = '#ef4444';
    } else if (textarea.value.length > 400) {
        charCount.style.color = '#f59e0b';
    } else {
        charCount.style.color = '#94a3b8';
    }
}

// Load model statistics
async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        document.getElementById('modelStatus').textContent = `${data.model.split('/')[1]} Ready`;
    } catch (error) {
        console.error('Error loading stats:', error);
        document.getElementById('modelStatus').textContent = 'Ready';
    }
}

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();
    
    const textarea = document.getElementById('textInput');
    const text = textarea.value.trim();
    
    if (!text) {
        showError('Please enter some text to predict emojis');
        return;
    }
    
    if (text.length < 3) {
        showError('Please enter at least 3 characters');
        return;
    }
    
    currentText = text;
    
    // Show loading state
    const predictBtn = document.getElementById('predictBtn');
    const originalContent = predictBtn.innerHTML;
    predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Predicting...</span>';
    predictBtn.classList.add('loading');
    predictBtn.disabled = true;
    
    // Hide previous results
    document.getElementById('resultSection').style.display = 'none';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text
            }),
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResult(data);
            // Scroll to result
            setTimeout(() => {
                document.getElementById('resultSection').scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'start' 
                });
            }, 100);
        } else {
            showError(data.error || 'Failed to predict emojis');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Network error. Please try again.');
    } finally {
        predictBtn.innerHTML = originalContent;
        predictBtn.classList.remove('loading');
        predictBtn.disabled = false;
    }
}

// Display prediction result
function displayResult(data) {
    predictedEmojis = data.emojis || [];
    
    // Set result text
    document.getElementById('resultText').textContent = data.text;
    
    // Set emoji count
    document.getElementById('emojiCount').textContent = data.count || 0;
    
    // Set generation time
    document.getElementById('resultTime').innerHTML = 
        `<i class="fas fa-clock"></i> ${data.generation_time}s`;
    
    // Display emojis
    const emojiDisplay = document.getElementById('emojiDisplay');
    emojiDisplay.innerHTML = '';
    
    if (predictedEmojis.length > 0) {
        predictedEmojis.forEach((emoji, index) => {
            const emojiSpan = document.createElement('span');
            emojiSpan.className = 'emoji-item';
            emojiSpan.textContent = emoji;
            emojiSpan.style.animationDelay = `${index * 0.1}s`;
            emojiSpan.title = `Click to copy: ${emoji}`;
            emojiSpan.onclick = () => copyEmoji(emoji);
            emojiDisplay.appendChild(emojiSpan);
        });
    } else {
        emojiDisplay.innerHTML = '<p style="color: var(--text-secondary);">No emojis predicted</p>';
    }
    
    // Show result section
    const resultSection = document.getElementById('resultSection');
    resultSection.style.display = 'block';
}

// Show error message
function showError(message) {
    alert('Error: ' + message);
}

// Copy single emoji
function copyEmoji(emoji) {
    copyToClipboard(emoji, `Copied ${emoji} to clipboard!`);
}

// Copy all emojis
function copyEmojis() {
    if (predictedEmojis.length === 0) {
        showError('No emojis to copy');
        return;
    }
    
    const emojisText = predictedEmojis.join(' ');
    copyToClipboard(emojisText, 'Emojis copied to clipboard!');
}

// Copy text with emojis
function copyWithText() {
    if (predictedEmojis.length === 0) {
        showError('No emojis to copy');
        return;
    }
    
    const emojisText = predictedEmojis.join(' ');
    const fullText = `${currentText} ${emojisText}`;
    copyToClipboard(fullText, 'Text with emojis copied!');
}

// Copy to clipboard with fallback
function copyToClipboard(text, successMessage) {
    // Try modern clipboard API first
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => {
            showCopySuccess(successMessage);
        }).catch(err => {
            console.error('Clipboard API failed:', err);
            fallbackCopy(text, successMessage);
        });
    } else {
        // Use fallback method
        fallbackCopy(text, successMessage);
    }
}

// Fallback copy method
function fallbackCopy(text, successMessage) {
    try {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.top = '-9999px';
        textarea.style.left = '-9999px';
        document.body.appendChild(textarea);
        
        textarea.focus();
        textarea.select();
        
        const successful = document.execCommand('copy');
        document.body.removeChild(textarea);
        
        if (successful) {
            showCopySuccess(successMessage);
        } else {
            throw new Error('execCommand failed');
        }
    } catch (err) {
        console.error('Fallback copy failed:', err);
        prompt('Copy this text manually:', text);
    }
}

// Show copy success message
function showCopySuccess(message) {
    // Create toast notification
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3);
        z-index: 1000;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.95rem;
        animation: slideInRight 0.3s ease;
    `;
    
    toast.innerHTML = `<i class="fas fa-check-circle"></i>${message}`;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    }, 2000);
}

// Predict again with same text
function predictAgain() {
    const form = document.getElementById('predictionForm');
    const textarea = document.getElementById('textInput');
    if (textarea.value.trim()) {
        handleSubmit(new Event('submit'));
    }
}

// Try example prompt
function tryExample(text) {
    document.getElementById('textInput').value = text;
    updateCharCount();
    
    // Scroll to textarea
    document.getElementById('textInput').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center' 
    });
    
    // Focus textarea
    document.getElementById('textInput').focus();
}

// Add CSS for toast animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
