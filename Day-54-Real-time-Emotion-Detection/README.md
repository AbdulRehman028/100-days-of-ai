# 🎭 Real-time Emotion Detection

A Flask web application that detects human emotions from webcam video or uploaded images using deep learning and computer vision.

## ✨ Features

- 📷 **Real-time Webcam Detection** - Capture emotions live from your webcam
- 🖼️ **Image Upload** - Upload photos for emotion analysis
- 🤖 **CNN-based Model** - Deep learning model for 7 emotion classes
- 👤 **Face Detection** - Automatic face detection using OpenCV Haar Cascade
- 📊 **Confidence Scores** - View probabilities for all emotions
- 🎨 **Modern UI** - Beautiful dark theme with Tailwind CSS

## 🎯 Emotions Detected

| Emotion | Emoji |
|---------|-------|
| Angry | 😠 |
| Disgust | 🤢 |
| Fear | 😨 |
| Happy | 😊 |
| Sad | 😢 |
| Surprise | 😲 |
| Neutral | 😐 |

## 🛠️ Installation

1. **Navigate to project directory:**
   ```bash
   cd Day-54-Real-time-Emotion-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

1. **Run the application:**
   ```bash
   python app.py
   ```

2. **Open in browser:**
   ```
   http://localhost:5000
   ```

3. **Choose input method:**
   - **Webcam**: Click "Start Camera" → "Capture" for single shots, or enable "Auto-detect" for continuous detection
   - **Upload**: Click or drag an image into the upload area → Click "Analyze Image"

## 🏗️ Architecture

```
Day-54-Real-time-Emotion-Detection/
├── app.py                  # Flask backend with CNN model
├── requirements.txt        # Python dependencies
├── README.md
├── models/                 # Saved model weights
│   └── emotion_model.h5
├── static/
│   ├── style.css          # Custom styles
│   └── script.js          # Webcam & UI logic
├── templates/
│   └── index.html         # Main UI template
└── uploads/               # Temporary upload storage
```

## 🧠 Model Details

- **Input**: 48×48 grayscale face images
- **Architecture**: 4-block CNN with batch normalization and dropout
- **Output**: 7-class softmax (emotions)
- **Face Detection**: OpenCV Haar Cascade Classifier

## 📊 Output

- Bounding boxes around detected faces
- Primary emotion with emoji and confidence score
- Bar chart showing all emotion probabilities
- Support for multiple faces in one image

## 🔧 Technical Stack

- **Backend**: Flask, TensorFlow/Keras
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Computer Vision**: OpenCV
- **Deep Learning**: Convolutional Neural Network (CNN)

## 💡 Tips for Best Results

- Ensure good, even lighting on your face
- Face the camera directly
- Remove glasses or face obstructions
- Use a neutral background

## 📝 Notes

- First run creates a demo model with random weights
- For production accuracy, train on FER2013 dataset or use pre-trained weights
- Auto-detect mode runs at ~2 FPS for smooth performance

## 🎯 Use Cases

- Emotion tracking for mental health apps
- Customer sentiment analysis
- Interactive gaming experiences
- Educational tools for emotion recognition
- Human-computer interaction research
