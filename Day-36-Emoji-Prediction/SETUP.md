# 🚀 Quick Setup Guide - Emoji Prediction

## ✅ Complete Setup (One-Time)

### Step 1: Navigate to Project
```powershell
cd "C:\my folder\100-days-of-ai\Day-36-Emoji-Prediction"
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

Packages installed:
- `flask` - Web framework
- `requests` - HTTP client for API calls
- `python-dotenv` - Environment variable loader

### Step 3: Setup Environment Variables

1. **Copy the example file:**
   ```powershell
   copy .env.example .env
   ```

2. **Get your HuggingFace token:**
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it (e.g., "Emoji Predictor")
   - Select "Read" access
   - Copy the token (starts with `hf_`)

3. **Edit `.env` file:**
   - Open `.env` in any text editor
   - Replace `YOUR_TOKEN_HERE` with your actual token:
   ```
   HF_API_TOKEN=hf_your_actual_token_here
   ```
   - Save the file

### Step 4: Run the App
```powershell
python app.py
```

You should see:
```
✅ HuggingFace token loaded from .env file!
💡 Using HuggingFace Router API
🤖 Model: Llama 3.2 3B Instruct
😀 Emoji Prediction Ready!
 * Running on http://127.0.0.1:5000
```

### Step 5: Open in Browser
Navigate to: **http://localhost:5000**

---

## 🎮 How to Use

1. **Enter text** in the input box
2. **Click "Predict Emojis"**
3. **View AI-suggested emojis**
4. **Click to copy** individual emojis
5. **Use buttons** to copy all or copy with text

---

## 📝 Example Texts to Try

```
I just got a promotion at work! So excited!
Feeling sad and lonely today
Going to the beach tomorrow! Can't wait!
I love pizza, burgers, and ice cream!
Happy birthday! Wishing you all the best!
My cat is so cute and fluffy!
```

---

## 🎯 Features

✅ **AI-Powered** - Uses Llama 3.2 3B LLM
✅ **1000+ Emojis** - Comprehensive emoji database
✅ **Context-Aware** - Understands emotions and topics
✅ **Fast** - 2-5 second predictions
✅ **Fallback System** - Keyword matching if LLM fails
✅ **Copy Options** - Copy emojis or text with emojis
✅ **Beautiful UI** - Dark theme with animations

---

## 🐛 Troubleshooting

**Problem:** "No HuggingFace API token found!"
- **Solution:** Make sure `.env` file exists with `HF_API_TOKEN=your_token`

**Problem:** Copy button doesn't work
- **Solution:** Use `http://localhost:5000` instead of `127.0.0.1`

**Problem:** No emojis predicted
- **Solution:** Try more descriptive text. System will use fallback keywords.

**Problem:** Slow first prediction
- **Solution:** First API call takes longer. Subsequent requests are faster.

---

## 🔒 Security

✅ `.env` file is gitignored (safe from GitHub)
✅ Token loads automatically
✅ No manual environment variables needed

---

## 📊 What It Does

1. **Analyzes your text** with AI
2. **Identifies emotions** (happy, sad, excited, etc.)
3. **Detects topics** (food, work, travel, etc.)
4. **Suggests emojis** that match the context
5. **Orders by relevance**
6. **Provides fallback** if AI fails

---

Enjoy predicting emojis! 😀✨
