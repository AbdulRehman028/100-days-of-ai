// Meme Caption Generator JavaScript

let currentDescription = '';
let currentCaptions = [];

document.addEventListener('DOMContentLoaded', function() {
    // Setup form submission
    const form = document.getElementById('captionForm');
    form.addEventListener('submit', handleSubmit);
    
    // Setup character counter
    const textarea = document.getElementById('description');
    textarea.addEventListener('input', updateCharCount);
    
    // Setup range slider
    const countSlider = document.getElementById('count');
    countSlider.addEventListener('input', updateCountValue);
    
    // Load model stats
    loadStats();
});

// Update character counter
function updateCharCount() {
    const textarea = document.getElementById('description');
    const charCount = document.getElementById('charCount');
    charCount.textContent = textarea.value.length;
    
    // Change color when approaching limit
    if (textarea.value.length > 450) {
        charCount.style.color = '#ff1744';
    } else if (textarea.value.length > 400) {
        charCount.style.color = '#ff9800';
    } else {
        charCount.style.color = '#9fa8da';
    }
}

// Update count value display
function updateCountValue() {
    const countSlider = document.getElementById('count');
    const countValue = document.getElementById('countValue');
    countValue.textContent = countSlider.value;
}

// Load model statistics
async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        const modelName = data.model.split('/')[1];
        document.getElementById('modelStatus').textContent = `${modelName} Ready`;
    } catch (error) {
        console.error('Error loading stats:', error);
        document.getElementById('modelStatus').textContent = 'Ready';
    }
}

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();
    
    const textarea = document.getElementById('description');
    const description = textarea.value.trim();
    const style = document.getElementById('style').value;
    const count = document.getElementById('count').value;
    
    if (!description) {
        showError('Please describe your meme image');
        return;
    }
    
    if (description.length < 5) {
        showError('Description is too short. Add more details!');
        return;
    }
    
    currentDescription = description;
    
    // Show loading state
    const generateBtn = document.getElementById('generateBtn');
    const originalContent = generateBtn.innerHTML;
    generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Generating...</span>';
    generateBtn.classList.add('loading');
    generateBtn.disabled = true;
    
    // Hide previous results
    document.getElementById('resultSection').style.display = 'none';
    
    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                description: description,
                style: style,
                count: parseInt(count)
            }),
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResult(data);
            // Scroll to result on mobile
            setTimeout(() => {
                const resultSection = document.getElementById('resultSection');
                if (window.innerWidth < 1024) {
                    resultSection.scrollIntoView({ 
                        behavior: 'smooth', 
                        block: 'start' 
                    });
                }
            }, 100);
        } else {
            showError(data.error || 'Failed to generate captions');
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

// Display generation result
function displayResult(data) {
    currentCaptions = data.captions || [];
    
    // Set result description
    document.getElementById('resultDescription').textContent = data.description;
    
    // Set caption count
    document.getElementById('captionCount').textContent = currentCaptions.length;
    
    // Set generation time
    document.getElementById('resultTime').innerHTML = 
        `<i class="fas fa-clock"></i> ${data.generation_time}s`;
    
    // Display captions
    const captionsList = document.getElementById('captionsList');
    captionsList.innerHTML = '';
    
    if (currentCaptions.length > 0) {
        currentCaptions.forEach((caption, index) => {
            const captionDiv = document.createElement('div');
            captionDiv.className = 'caption-item';
            captionDiv.style.animationDelay = `${index * 0.1}s`;
            captionDiv.onclick = () => copyCaption(caption, index);
            
            captionDiv.innerHTML = `
                <span class="caption-number">${index + 1}</span>
                <span class="caption-text">${caption}</span>
                <i class="fas fa-copy copy-icon"></i>
            `;
            
            captionsList.appendChild(captionDiv);
        });
    } else {
        captionsList.innerHTML = '<p style="color: var(--text-secondary); text-align: center;">No captions generated</p>';
    }
    
    // Show result section
    const resultSection = document.getElementById('resultSection');
    resultSection.style.display = 'block';
}

// Copy single caption
function copyCaption(caption, index) {
    copyToClipboard(caption, `Caption ${index + 1} copied!`);
}

// Copy all captions
function copyAllCaptions() {
    if (currentCaptions.length === 0) {
        showError('No captions to copy');
        return;
    }
    
    const allCaptions = currentCaptions
        .map((caption, index) => `${index + 1}. ${caption}`)
        .join('\n\n');
    
    copyToClipboard(allCaptions, 'All captions copied!');
}

// Download captions as text file
function downloadCaptions() {
    if (currentCaptions.length === 0) {
        showError('No captions to download');
        return;
    }
    
    const content = `Meme Captions\n${'='.repeat(50)}\n\n` +
                   `Image Description:\n${currentDescription}\n\n` +
                   `Generated Captions:\n${'-'.repeat(50)}\n\n` +
                   currentCaptions.map((caption, index) => `${index + 1}. ${caption}`).join('\n\n') +
                   `\n\n${'='.repeat(50)}\n` +
                   `Generated by Meme Caption Generator\n` +
                   `Date: ${new Date().toLocaleString()}`;
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `meme-captions-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showSuccess('Captions downloaded!');
}

// Generate again (clear and focus on input)
function generateAgain() {
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('description').focus();
}

// Try example description
function tryExample(description) {
    const textarea = document.getElementById('description');
    textarea.value = description;
    updateCharCount();
    textarea.focus();
    
    // Scroll to textarea
    textarea.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Copy to clipboard with fallback
async function copyToClipboard(text, message) {
    try {
        // Modern clipboard API
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
            showSuccess(message);
        } else {
            // Fallback for older browsers
            fallbackCopy(text);
            showSuccess(message);
        }
    } catch (error) {
        console.error('Copy failed:', error);
        fallbackCopy(text);
        showSuccess(message);
    }
}

// Fallback copy method
function fallbackCopy(text) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    
    try {
        document.execCommand('copy');
    } catch (error) {
        console.error('Fallback copy failed:', error);
    }
    
    document.body.removeChild(textarea);
}

// Show success toast
function showSuccess(message) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
        <i class="fas fa-check-circle"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        document.body.removeChild(toast);
    }, 3000);
}

// Show error message
function showError(message) {
    alert('Error: ' + message);
}