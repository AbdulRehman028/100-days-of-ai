# Day 53: Text-to-Speech Notes Maker 🎙️

An AI-powered application that generates comprehensive notes on any topic using GPT-2 and converts them to speech using Google Text-to-Speech (gTTS).

## 🌟 Features

- **AI Note Generation**: Uses GPT-2 to generate notes on any topic
- **Multiple Note Styles**:
  - 📖 Detailed - Comprehensive explanations
  - • Bullet Points - Key points format
  - 📋 Summary - Brief overview
  - 📚 Study Notes - Learning-focused format
- **Adjustable Length**: Short, Medium, or Long notes
- **Multi-language TTS**: Support for 12+ languages
- **Speed Control**: Normal or slower speech option
- **Instant Playback**: Listen to notes directly in browser
- **Download MP3**: Save audio files for offline listening
- **Modern UI**: Beautiful glass-morphism design with progress timeline

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **LLM**: GPT-2 (Hugging Face Transformers)
- **TTS**: gTTS (Google Text-to-Speech)
- **Frontend**: HTML, TailwindCSS, JavaScript

## 📦 Installation

1. Navigate to project directory:
```bash
cd Day-53-Text-to-Speech-Notes-Maker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open http://localhost:5000 in your browser

## 🎯 How to Use

1. **Enter Topic**: Type any topic you want notes on (e.g., "Machine Learning", "Climate Change")
2. **Choose Style**: Select from Detailed, Bullet Points, Summary, or Study Notes
3. **Set Length**: Use slider to choose Short, Medium, or Long
4. **Select Language**: Choose from 12+ languages for audio
5. **Generate**: Click "Generate Notes & Audio" button
6. **Listen/Download**: Play audio in browser or download MP3

## 🌐 Supported Languages

- 🇺🇸 English (US)
- 🇬🇧 English (UK)
- 🇦🇺 English (AU)
- 🇪🇸 Spanish
- 🇫🇷 French
- 🇩🇪 German
- 🇮🇹 Italian
- 🇵🇹 Portuguese
- 🇯🇵 Japanese
- 🇰🇷 Korean
- 🇨🇳 Chinese
- 🇮🇳 Hindi

## 📁 Project Structure

```
Day-53-Text-to-Speech-Notes-Maker/
├── app.py                 # Flask backend
├── requirements.txt       # Dependencies
├── README.md             # Documentation
├── templates/
│   └── index.html        # Frontend UI
└── static/
    └── audio/            # Generated audio files
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main page |
| `/generate-notes` | POST | Generate notes only |
| `/convert-to-speech` | POST | Convert text to speech |
| `/generate-and-speak` | POST | Generate notes + TTS |
| `/download/<filename>` | GET | Download audio file |

## 💡 Example Topics

- Artificial Intelligence
- Climate Change
- Quantum Computing
- The Solar System
- Human Brain
- Blockchain Technology
- Renaissance Art
- Economic Theory

## 🚀 Day 53 of 100 Days of AI

This project demonstrates the integration of:
- Large Language Models (LLM) for text generation
- Text-to-Speech (TTS) for audio conversion
- End-to-end AI pipeline with beautiful UI
