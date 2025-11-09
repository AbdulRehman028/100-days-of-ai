# 🚀 Quick Setup Guide

## ✅ Complete Setup (One-Time)

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

This installs:
- `flask` - Web framework
- `requests` - HTTP client for API calls
- `python-dotenv` - Environment variable loader

### Step 2: Setup Environment Variables

1. **Copy the example file:**
   ```powershell
   copy .env.example .env
   ```

2. **Get your HuggingFace token:**
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it (e.g., "Text Generator")
   - Select "Read" access
   - Click "Generate"
   - Copy the token (starts with `hf_`)

3. **Edit `.env` file:**
   - Open `.env` in any text editor
   - Replace `YOUR_TOKEN_HERE` with your actual token:
   ```
   HF_API_TOKEN=hf_your_actual_token_here
   ```
   - Save the file

### Step 3: Run the App
```powershell
python app.py
```

You should see:
```
✅ HuggingFace token loaded from .env file!
💡 Using NEW Router API (OpenAI-compatible)
🤖 Model: Llama 3.2 3B Instruct (Meta's latest!)
🆓 Free API - No downloads!
 * Running on http://127.0.0.1:5000
```

### Step 4: Open in Browser
Navigate to: **http://localhost:5000**

---

## 🔒 Security Notes

✅ **Your `.env` file is safe:**
- Added to `.gitignore` (won't be committed to GitHub)
- Keeps your API token secure
- Not shared publicly

✅ **Token persists:**
- No need to set it every time
- Automatically loaded on app startup
- Works across terminal sessions

❌ **Never commit `.env` to GitHub:**
- The `.gitignore` file prevents this
- Share `.env.example` instead (without real token)

---

## 🎯 Usage

1. Select content type (Story, Poem, Social Post, etc.)
2. Enter your prompt
3. Adjust settings (optional)
4. Click "Generate with AI"
5. Copy or download your content!

---

## 🐛 Troubleshooting

**Problem:** "WARNING: No HuggingFace API token found!"
- **Solution:** Make sure `.env` file exists and contains `HF_API_TOKEN=your_token`

**Problem:** Copy button doesn't work
- **Solution:** Use `http://localhost:5000` instead of `http://127.0.0.1:5000`

**Problem:** Generation fails with 401 error
- **Solution:** Your token is invalid. Get a new token from HuggingFace

**Problem:** Stories are incomplete
- **Solution:** Increase the length slider to 1500-2500 tokens

---

## 📚 What You Can Generate

### 8 Content Types:
1. 📖 **Story** - Epic narratives with detail
2. 🎭 **Poem** - Beautiful verses
3. 🎬 **Script** - Movie/play scenes
4. 🗺️ **Quest** - Adventure scenarios
5. 📱 **Social Post** - Engaging social media content
6. 📝 **Blog** - Comprehensive blog articles
7. ✉️ **Email** - Professional emails
8. 📰 **Article** - Informative articles

Enjoy creating! 🎉
