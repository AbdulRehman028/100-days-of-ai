# � Day 33 — AI Text Generator (Stories, Poems, Scripts & Quests)

### 🎯 Goal

Build a beautiful modern **web application** that uses the **GPT-2 AI model** to generate creative stories, poems, movie scripts, and adventure quests - all for free!

---

## ✨ Features

### 🎨 Modern Web Interface

- 🌈 **Beautiful dark theme** with gradient animations
- ✨ **Particle background** effects
- 📝 **4 Content Types**: Stories, Poems, Scripts, Quests
- 🎛️ **Advanced settings**: Temperature & length controls
- ⚡ **Real-time generation** with loading animations
- 📱 **Fully responsive** design
- 💾 **Copy & Download** generated content

### � AI Capabilities

- 🧠 **GPT-2 Model** (124M parameters)
- 🆓 **100% Free & Open Source**
- � **No API keys** required
- 💻 **Runs locally** on your machine
- 🎨 **Creative & diverse** outputs
- 🎯 **Context-aware** generation

---

## 🧩 Tech Stack

- **Backend:** Python, Flask
- **AI Model:** HuggingFace Transformers 
- **ML Framework:** PyTorch
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **No external APIs** - Everything runs locally!

---

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Setup API Token

Create a `.env` file in the project directory:

```powershell
# Copy the example file
copy .env.example .env
```

Then edit `.env` and add your HuggingFace token:

```
HF_API_TOKEN=hf_your_actual_token_here
```

**Get your FREE token:** https://huggingface.co/settings/tokens

### 3. Run the Web App

```powershell
python app.py
```

✅ **No downloads needed** - Uses HuggingFace API (cloud-based)!

### 4. Open in Browser

Navigate to **http://127.0.0.1:5000** or **http://localhost:5000**

---

## 🎮 How to Use

### Creating Content:

1. **Choose Content Type**

   - 📖 Story - Creative narratives
   - 🎭 Poem - Artistic verses
   - 🎬 Script - Movie/play scripts
   - 🗺️ Quest - Adventure scenarios
2. **Enter Your Prompt**

   - Type your idea or starting text
   - Be specific for better results
   - Minimum 3 characters
3. **Adjust Settings (Optional)**

   - **Creativity Level** (Temperature): 0.3-1.5
     - Lower = More focused & coherent
     - Higher = More creative & random
   - **Length**: 100-500 tokens
4. **Generate!**

   - Click "Generate with AI"
   - Wait 2-5 seconds
   - View your generated content
5. **Copy or Download**

   - Copy to clipboard
   - Download as .txt file
   - Generate again with same settings

---

## 📊 Example Prompts

### 📖 **Story Examples:**

- "A brave knight embarks on a quest to save the kingdom"
- "A mysterious stranger arrives in a small town"
- "In a world where magic is forbidden..."

### 🎭 **Poem Examples:**

- "The moon whispers secrets to the stars"
- "Autumn leaves dancing in the wind"
- "A lonely lighthouse stands guard"

### 🎬 **Script Examples:**

- "INT. SPACESHIP - A captain discovers an alien artifact"
- "EXT. MEDIEVAL CASTLE - DAY - A messenger arrives"
- "INT. DETECTIVE'S OFFICE - NIGHT"

### 🗺️ **Quest Examples:**

- "Find the ancient treasure hidden in the enchanted forest"
- "Rescue the princess from the dragon's lair"
- "Discover the secret of the lost temple"
