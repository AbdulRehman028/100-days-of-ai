# Day 39: Denoising Autoencoder for Text/Images

![Denoising Autoencoder](https://img.shields.io/badge/Deep%20Learning-Autoencoder-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Flask](https://img.shields.io/badge/Flask-3.0-green)

## 🎯 Project Overview

A comprehensive end-to-end deep learning project that implements denoising autoencoders for both images and text. The project features a beautiful web interface where users can:

- **Image Denoising**: Remove noise from handwritten digits using a Convolutional Autoencoder trained on MNIST
- **Text Denoising**: Recover corrupted text using an LSTM-based Autoencoder
- **Real-time Evaluation**: View metrics like MSE, accuracy, and improvement percentages
- **Interactive Interface**: Draw digits or upload images, enter text with adjustable noise levels.

## 🚀 Features

### Image Denoising
- ✅ Convolutional Autoencoder (Conv2D + MaxPooling + UpSampling)
- ✅ Trained on MNIST dataset (60,000 training images)
- ✅ Draw digits on canvas or upload images
- ✅ Adjustable noise levels (0.1 - 0.8)
- ✅ Real-time MSE and improvement metrics
- ✅ Visual comparison (Original → Noisy → Denoised)

### Text Denoising
- ✅ LSTM Autoencoder for character-level denoising
- ✅ Handles lowercase, digits, punctuation, and spaces
- ✅ Quick example templates
- ✅ Adjustable noise levels (0.1 - 0.5)
- ✅ Character-level accuracy metrics
- ✅ Side-by-side text comparison

### Modern UI/UX
- ✅ Beautiful Tailwind CSS design
- ✅ Responsive layout (mobile, tablet, desktop)
- ✅ Animated gradients and effects
- ✅ Tab-based interface (Image/Text modes)
- ✅ Real-time model status indicators
- ✅ Toast notifications

## 📋 Requirements

```
flask==3.0.0
tensorflow==2.15.0
numpy==1.24.3
pillow==10.1.0
werkzeug==3.0.1
```

## 🛠️ Installation

1. **Clone or navigate to the project directory:**
```bash
cd Day-39-Denoising-Autoencoder
```

2. **Activate virtual environment:**
```bash
# Windows
..\venv\Scripts\activate

# Mac/Linux
source ../venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🎮 Usage

1. **Start the Flask server:**
```bash
python app.py
```

2. **Open browser and navigate to:**
```
http://127.0.0.1:5000
```

3. **First Launch:**
   - Models will train automatically on first run
   - Image model: ~2-3 minutes (10 epochs on MNIST)
   - Text model: ~3-5 minutes (20 epochs)
   - Models are saved and loaded on subsequent runs

4. **Image Denoising:**
   - Switch to "Image" tab
   - Draw a digit (0-9) on the canvas OR upload an image
   - Adjust noise level slider (default: 0.3)
   - Click "Denoise Image"
   - View Original → Noisy → Denoised comparison
   - Check MSE metrics and improvement percentage

5. **Text Denoising:**
   - Switch to "Text" tab
   - Enter text (max 50 characters) or use quick examples
   - Adjust noise level slider (default: 0.2)
   - Click "Denoise Text"
   - View Original → Noisy → Denoised comparison
   - Check accuracy metrics

## 🧠 Model Architecture

### Image Autoencoder (Convolutional)
```
Encoder:
- Input: (28, 28, 1)
- Conv2D(32) → MaxPool
- Conv2D(64) → MaxPool
- Conv2D(128)

Decoder:
- Conv2D(128)
- UpSample → Conv2D(64)
- UpSample → Conv2D(32)
- Output: Conv2D(1) with sigmoid
```

**Training:**
- Dataset: MNIST (60,000 train, 10,000 test)
- Noise: Gaussian noise (factor: 0.5)
- Optimizer: Adam
- Loss: MSE
- Epochs: 10
- Batch size: 128

### Text Autoencoder (LSTM)
```
Encoder:
- Input: (50,) sequence
- Embedding(128)
- LSTM(256) → LSTM(128) → LSTM(64)

Decoder:
- RepeatVector(50)
- LSTM(64) → LSTM(128) → LSTM(256)
- TimeDistributed Dense(vocab_size) with softmax
```

**Training:**
- Dataset: 5,000 generated samples
- Vocabulary: a-z, 0-9, punctuation, space (~100 chars)
- Noise: Random character replacement (20%)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 20
- Batch size: 64

## 📊 Evaluation Metrics

### Image Denoising
- **MSE (Noisy)**: Mean Squared Error between original and noisy images
- **MSE (Denoised)**: Mean Squared Error between original and denoised images
- **Improvement**: Percentage improvement in MSE

### Text Denoising
- **Accuracy**: Character-level accuracy (% of correctly recovered characters)
- **Noise Level**: Percentage of corrupted characters

## 🎨 Project Structure

```
Day-39-Denoising-Autoencoder/
├── app.py                 # Flask backend with autoencoder models
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── script.js         # Frontend JavaScript
│   └── style.css         # Custom CSS animations
├── models/               # Saved model files
│   ├── image_autoencoder.h5
│   ├── text_autoencoder.h5
│   └── text_vocab.npy
└── uploads/             # Temporary upload directory
```

## 🔬 Technical Details

### Noise Addition

**Images:**
- Gaussian noise: `noisy = image + noise_factor * N(0, 1)`
- Clipped to [0, 1] range

**Text:**
- Random character replacement
- Each character has `noise_level` probability of being replaced
- Replacement characters drawn from vocabulary

### Performance

- **Image Model Size**: ~2.5 MB
- **Text Model Size**: ~15 MB
- **Inference Time (Image)**: ~50ms
- **Inference Time (Text)**: ~100ms
- **GPU Support**: Yes (if TensorFlow GPU available)

## 🐛 Troubleshooting

### Issue: Models not training
**Solution**: Models train automatically on first run. Wait 5-10 minutes for both models to complete training.

### Issue: TensorFlow warnings
**Solution**: TensorFlow may show optimization warnings. These are normal and don't affect functionality.

### Issue: Out of memory
**Solution**: Reduce batch size in `app.py` (lines for `batch_size` parameter).

### Issue: Image not recognized
**Solution**: Draw clearly with bold strokes. The model is trained on MNIST digits (0-9).

## 🎓 Learning Outcomes

This project demonstrates:
1. ✅ Building and training autoencoders
2. ✅ Convolutional layers for image processing
3. ✅ LSTM layers for sequence processing
4. ✅ Adding controlled noise for training
5. ✅ Model evaluation with metrics
6. ✅ Flask integration with TensorFlow
7. ✅ Real-time inference
8. ✅ Beautiful UI/UX design

## 🔮 Future Enhancements

- [ ] Add more noise types (salt & pepper, speckle)
- [ ] Support for color images
- [ ] Longer text sequences
- [ ] Multiple language support
- [ ] Model comparison dashboard
- [ ] Export denoised results
- [ ] Batch processing

## 📝 Notes

- Models are automatically saved after training
- Pre-trained models are loaded on subsequent runs
- Canvas drawings are converted to 28x28 grayscale
- Text is limited to 50 characters for optimal performance

## 🙏 Acknowledgments

- MNIST dataset from Yann LeCun
- TensorFlow/Keras framework
- Tailwind CSS for beautiful UI

---

**Built with ❤️ by Abdur Rehman Baig**  
*Day 39 of 100 Days of AI Challenge*
