# 😂 Meme Caption Generator

AI-powered meme caption generator using Llama 3.2 LLM. Generate creative, funny, and viral-ready captions for your memes!

## ✨ Features

- 🤖 **AI-Powered**: Uses Llama 3.2 3B LLM for intelligent caption generation
- 🎨 **8 Caption Styles**: Funny, Relatable, Sarcastic, Motivational, Dark Humor, Wholesome, Absurd, Pop Culture
- ⚡ **Fast Generation**: Get 1-10 caption options in seconds
- 📋 **Easy Copy**: Click any caption to copy it instantly
- 💾 **Download**: Save all captions as a text file
- 🔥 **Viral Ready**: Captions designed to be shareable and engaging
- 📱 **Responsive**: Works perfectly on desktop and mobile

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- HuggingFace API Token (free at https://huggingface.co/settings/tokens)

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Setup environment variables:**
```bash
# Copy example file
copy .env.example .env

# Edit .env and add your HuggingFace token
HF_API_TOKEN=your_token_here
```

3. **Run the app:**
```bash
python app.py
```

4. **Open in browser:**
```
http://localhost:5000
```

## 🎮 How to Use

1. **Describe your meme image** (e.g., "A cat looking confused at a computer")
2. **Select a caption style** (Funny, Sarcastic, etc.)
3. **Choose number of captions** (1-10)
4. **Click "Generate Captions"**
5. **Copy individual captions** or download all
6. **Use in your memes!** 😂

## 📝 Example Descriptions

```
"A dog staring at a laptop looking stressed"
"Person running away while being chased by responsibilities"
"Drake refusing one thing and pointing at another"
"Cat sitting at desk with coffee looking tired"
"Person checking bank account with horrified expression"
"Two buttons: one says sleep, one says stay up"
```

## 🎨 Caption Styles

| Style | Description | Best For |
|-------|-------------|----------|
| 😂 Funny | Classic humor | General memes |
| 😌 Relatable | Everyday situations | Life memes |
| 😏 Sarcastic | Witty & clever | Edgy content |
| 💪 Motivational | Inspirational | Positive vibes |
| 💀 Dark Humor | Edgy comedy | Adult humor |
| 🥰 Wholesome | Heartwarming | Feel-good content |
| 🤪 Absurd | Surreal & random | Weird memes |
| 🎬 Pop Culture | Trendy references | Current events |

## 📊 Example Output

**Description:** "A cat staring at a laptop looking confused"

**Generated Captions (Funny Style):**
1. When you realize it's Monday tomorrow
2. Me trying to understand my life choices
3. POV: You're checking your exam results
4. My brain during an important meeting
5. When someone asks "What are your plans for the future?"

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **AI Model**: Llama 3.2 3B Instruct
- **API**: HuggingFace Router API
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Styling**: Custom CSS with animations
- **Icons**: Font Awesome 6.4.0

## 📂 Project Structure

```
Day-37-Meme-Caption-Generator/
├── app.py                 # Flask backend
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── .env.example          # Environment template
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── static/
│   ├── style.css        # Styling
│   └── script.js        # Frontend logic
├── templates/
│   └── index.html       # Main interface
└── uploads/             # Upload directory
```

## 🔧 Configuration

### Environment Variables
```env
HF_API_TOKEN=your_huggingface_token_here
```

### API Settings (in app.py)
- **Model**: `meta-llama/Llama-3.2-3B-Instruct`
- **Max Tokens**: 500
- **Temperature**: 0.9 (creative)
- **Top P**: 0.95
- **Timeout**: 30 seconds

## 🎯 Features Breakdown

### Caption Generation
- Uses LLM with specialized prompts for each style
- Parses numbered captions from response
- Fallback captions if parsing fails
- Removes duplicates and cleans formatting

### User Interface
- Two-column layout on large screens
- Sticky result section
- Animated emoji background
- Character counter with color coding
- Range slider for caption count
- Example prompts for quick testing
- Click-to-copy functionality
- Download as text file

### Copy Functionality
- Modern Clipboard API
- Fallback for older browsers
- Toast notifications
- Copy individual or all captions

## 🐛 Troubleshooting

**Problem:** "No HuggingFace API token found!"
- **Solution**: Create `.env` file with `HF_API_TOKEN=your_token`

**Problem:** Captions not generating
- **Solution**: Check internet connection and API token validity

**Problem:** Copy button doesn't work
- **Solution**: Use HTTPS or localhost (required for Clipboard API)

**Problem:** Slow generation
- **Solution**: First API call takes longer, subsequent calls are faster

## 🚀 Deployment

### Render.com
1. Create new Web Service
2. Connect GitHub repository
3. Add environment variable: `HF_API_TOKEN`
4. Deploy!

### HuggingFace Spaces
1. Create new Space (Gradio/Streamlit)
2. Upload all files
3. Add secret: `HF_API_TOKEN`
4. Space will auto-deploy

## 📈 Future Enhancements

- [ ] Image upload support
- [ ] Caption history
- [ ] Favorite captions
- [ ] Share to social media
- [ ] More caption styles
- [ ] Multi-language support
- [ ] Caption voting system
- [ ] Meme templates library

## 🤝 Contributing

This is a personal learning project, but suggestions are welcome!

## 📄 License

MIT License - feel free to use for learning

## 👨‍💻 Author

**Abdur Rehman Baig**
- Day 37 of 100 Days of AI Challenge
- Built with 😂 and lots of coffee ☕

## 🙏 Acknowledgments

- HuggingFace for the amazing API
- Meta for Llama 3.2 model
- The meme community for inspiration

---

**Made with ❤️ for the Day 37 of 100 Days of AI Challenge**

*Keep calm and meme on!* 😂🚀
