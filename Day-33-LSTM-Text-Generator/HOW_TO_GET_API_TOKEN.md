# 🔑 How to Get Your FREE HuggingFace API Token

## Step 1: Create HuggingFace Account (Free)
1. Go to: **https://huggingface.co/join**
2. Sign up with email or GitHub
3. Verify your email

## Step 2: Generate API Token
1. Go to: **https://huggingface.co/settings/tokens**
2. Click **"New token"**
3. Name it: `text-generator` (or any name)
4. Select Type: **"Read"** (default)
5. Click **"Generate a token"**
6. **Copy the token** (starts with `hf_...`)

## Step 3: Use the Token

### Option A: Environment Variable (Recommended)
**PowerShell:**
```powershell
$env:HF_API_TOKEN="hf_YOUR_TOKEN_HERE"
python app.py
```

**OR run both at once:**
```powershell
$env:HF_API_TOKEN="hf_YOUR_TOKEN_HERE"; python app.py
```

### Option B: Hardcode in app.py (NOT for GitHub!)
Open `app.py` and edit line 11:
```python
API_TOKEN = "hf_YOUR_TOKEN_HERE"  # Replace with your actual token
```

## Step 4: Run the App
```powershell
cd "c:\my folder\100-days-of-ai\Day-33-LSTM-Text-Generator"
python app.py
```

## 🎉 That's It!
Your app will now use the powerful **Mistral-7B-v0.3** model for FREE!

## 💡 Benefits:
- ✅ **Zero local storage** - no downloads
- ✅ **Professional quality** - 7 billion parameters
- ✅ **100% FREE** - no credit card needed
- ✅ **GitHub-friendly** - token stays secret
- ✅ **No rate limits** - generous free tier

## ⚠️ Security Tips:
1. **Never commit your token to GitHub!**
2. Add `.env` file to `.gitignore` if using dotenv
3. Use environment variables for production
4. Regenerate token if accidentally exposed

## 🔗 Useful Links:
- Token Settings: https://huggingface.co/settings/tokens
- Model Page: https://huggingface.co/mistralai/Mistral-7B-v0.3
- HuggingFace Docs: https://huggingface.co/docs/api-inference/index
