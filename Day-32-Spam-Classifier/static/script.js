// Spam Classifier JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Load model stats
    loadStats();
    
    // Setup form submission
    const form = document.getElementById('spamForm');
    form.addEventListener('submit', handleSubmit);
    
    // Setup character counter
    const textarea = document.getElementById('messageInput');
    textarea.addEventListener('input', updateCharCount);
});

// Load and display model statistics
async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        document.getElementById('accuracy').textContent = `${data.accuracy}% Accurate`;
    } catch (error) {
        console.error('Error loading stats:', error);
        document.getElementById('accuracy').textContent = 'Ready';
    }
}

// Update character counter
function updateCharCount() {
    const textarea = document.getElementById('messageInput');
    const charCount = document.getElementById('charCount');
    charCount.textContent = textarea.value.length;
}

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();
    
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) {
        showError('Please enter a message to analyze');
        return;
    }
    
    // Show loading state
    const analyzeBtn = document.getElementById('analyzeBtn');
    const originalContent = analyzeBtn.innerHTML;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Analyzing...</span>';
    analyzeBtn.classList.add('loading');
    analyzeBtn.disabled = true;
    
    try {
        const response = await fetch('/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Classification failed');
        }
        
        const result = await response.json();
        displayResult(result);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred. Please try again.');
    } finally {
        // Reset button
        analyzeBtn.innerHTML = originalContent;
        analyzeBtn.classList.remove('loading');
        analyzeBtn.disabled = false;
    }
}

// Display classification result
function displayResult(result) {
    const resultSection = document.getElementById('resultSection');
    const resultCard = document.getElementById('resultCard');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultMessage = document.getElementById('resultMessage');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const resultDetails = document.getElementById('resultDetails');
    
    // Remove previous classes
    resultCard.classList.remove('spam', 'legitimate');
    
    if (result.is_spam) {
        // Spam result
        resultCard.classList.add('spam');
        resultIcon.innerHTML = '<i class="fas fa-exclamation-triangle" style="color: var(--danger-color);"></i>';
        resultTitle.textContent = '🚨 SPAM DETECTED!';
        resultMessage.textContent = 'This message appears to be spam or unwanted content.';
        resultDetails.innerHTML = `
            <i class="fas fa-shield-alt" style="color: var(--danger-color);"></i>
            <strong style="color: var(--danger-color);">Warning:</strong> Be cautious! Do not click links or respond.
        `;
    } else {
        // Legitimate result
        resultCard.classList.add('legitimate');
        resultIcon.innerHTML = '<i class="fas fa-check-circle" style="color: var(--success-color);"></i>';
        resultTitle.textContent = '✅ LEGITIMATE MESSAGE';
        resultMessage.textContent = 'This message appears to be safe and genuine.';
        resultDetails.innerHTML = `
            <i class="fas fa-thumbs-up" style="color: var(--success-color);"></i>
            <strong style="color: var(--success-color);">Safe:</strong> This looks like a legitimate message.
        `;
    }
    
    // Update confidence
    confidenceValue.textContent = `${result.confidence}%`;
    confidenceFill.style.width = `${result.confidence}%`;
    
    // Show result section with animation
    resultSection.style.display = 'block';
    
    // Scroll to result
    setTimeout(() => {
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

// Show error message
function showError(message) {
    alert('⚠️ ' + message);
}

// Try example message
function tryExample(message) {
    const textarea = document.getElementById('messageInput');
    textarea.value = message;
    updateCharCount();
    
    // Scroll to textarea
    textarea.scrollIntoView({ behavior: 'smooth', block: 'center' });
    textarea.focus();
}
