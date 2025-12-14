# ✍️ Handwriting to Text Generator

AI-powered handwriting recognition system using Llama 3.2 LLM. Draw or upload handwritten images and convert them to digital text!

## ✨ Features

- 🎨 **Drawing Canvas**: Write with adjustable pen size and colors
- 📸 **Image Upload**: Support for PNG, JPG, JPEG, GIF, BMP
- 🤖 **AI Recognition**: Powered by Llama 3.2 3B Instruct
- ⚡ **Real-time Processing**: Fast text recognition
- 📊 **Detailed Stats**: Word count, line count, confidence score
- 💾 **Export Options**: Copy to clipboard or download as text
- 📱 **Responsive Design**: Works on desktop and mobile
- 🎨 **Modern UI**: Built with Tailwind CSS

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- HuggingFace API Token

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Setup environment:**
```bash
copy .env.example .env
# Add your HuggingFace token to .env
```

3. **Run the app:**
```bash
python app.py
```

4. **Open browser:**
```
http://localhost:5000
```

## 🎮 How to Use

### Drawing Mode
1. Click the **Draw** tab
2. Use the canvas to write your text
3. Adjust pen size and color as needed
4. Click **Recognize Text**
5. View the recognized text with stats

### Upload Mode
1. Click the **Upload** tab
2. Drag & drop an image or browse files
3. Click **Recognize Text**
4. Copy or download the result

## 📝 Example Use Cases

- ✍️ Convert handwritten notes to digital text
- 📋 Digitize paper forms and documents
- ✒️ Extract text from signatures
- 📝 Process handwritten lists and reminders
- 💌 Convert handwritten letters to text
- 📄 Archive historical handwritten documents

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **AI Model**: Llama 3.2 3B Instruct
- **API**: HuggingFace Router
- **Frontend**: HTML5 Canvas, JavaScript
- **Styling**: Tailwind CSS
- **Icons**: Font Awesome

## 📂 Project Structure

```
Day-38-Handwriting-to-Text/
├── app.py                # Flask backend
├── requirements.txt      # Dependencies
├── .env                 # Environment variables
├── .env.example         # Env template
├── .gitignore          # Git ignore
├── README.md           # Documentation
├── static/
│   ├── style.css       # Custom CSS
│   └── script.js       # Frontend logic
├── templates/
│   └── index.html      # Main interface
└── uploads/            # Temp upload folder
```

## 🎨 Features Breakdown

### Drawing Canvas
- Adjustable pen size (1-20px)
- Custom color picker
- Clear canvas function
- Touch support for mobile
- Smooth drawing experience

### Image Upload
- Drag & drop interface
- Multiple format support
- Image preview before processing
- File size validation (16MB max)

### Recognition System
- LLM-based text recognition
- Confidence scoring
- Processing time tracking
- Word, line, and character counting

### User Interface
- Modern gradient background with animated blobs
- Smooth animations and transitions
- Tab-based navigation
- Responsive grid layout
- Custom scrollbars
- Toast notifications

## 📊 Stats Displayed

| Metric | Description |
|--------|-------------|
| Words | Total word count |
| Lines | Number of lines |
| Chars | Character count |
| Score | Confidence percentage |

## 🎯 Example Scenarios

**Personal Notes:**
```
"Remember to buy milk!"
"Meeting at 3 PM"
```

**Signatures:**
```
"John Doe"
"Best regards, Sarah"
```

**Lists:**
```
1. Bread
2. Eggs
3. Butter
4. Milk
```

**Quotes:**
```
"Success is not final,
failure is not fatal:
it is the courage to continue
that counts."
```

## 🔧 Configuration

### Environment Variables
```env
HF_API_TOKEN=your_huggingface_token
```

### API Settings
- **Model**: `meta-llama/Llama-3.2-3B-Instruct`
- **Max Tokens**: 300
- **Temperature**: 0.7
- **Max File Size**: 16MB

## 🐛 Troubleshooting

**Canvas not drawing:**
- Check browser compatibility
- Try refreshing the page
- Ensure JavaScript is enabled

**Upload not working:**
- Verify file format (PNG, JPG, JPEG, GIF, BMP)
- Check file size (max 16MB)
- Try a different image

**Recognition errors:**
- Ensure API token is valid
- Check internet connection
- Try with clearer handwriting

**Copy button not working:**
- Use HTTPS or localhost
- Try the download option instead
- Check browser permissions

## 🚀 Deployment

### Render.com
```bash
1. Create new Web Service
2. Connect repository
3. Add HF_API_TOKEN env variable
4. Deploy
```

### Heroku
```bash
1. Create new app
2. Connect GitHub repo
3. Add Config Vars: HF_API_TOKEN
4. Deploy branch
```

## 📈 Future Enhancements

- [ ] Multiple language support
- [ ] Batch image processing
- [ ] History of recognized texts
- [ ] Advanced image preprocessing
- [ ] PDF export option
- [ ] Real-time recognition while drawing
- [ ] Handwriting style analysis
- [ ] Integration with cloud storage

## 🤝 Contributing

Personal learning project - suggestions welcome!

## 📄 License

MIT License

## 👨‍💻 Author

**Abdur Rehman Baig**
- Day 38 of 100 Days of AI Challenge
- Built with ✍️ and AI magic

## 🙏 Acknowledgments

- HuggingFace for the API
- Meta for Llama 3.2
- Tailwind CSS for styling
- Font Awesome for icons

---

**Made with ❤️ for Day 38 of 100 Days of AI**

*Converting handwriting to text, one stroke at a time!* ✍️✨
