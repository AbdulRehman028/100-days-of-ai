# 🛡️ Day 32 — AI Spam Classifier (CLI + Web Interface)

### 🎯 Goal

Build an **AI-powered Spam/Ham Classifier** with both a beautiful CLI interface and a modern web application using Naive Bayes and text vectorization.

---

## ✨ Features

### 🎨 CLI Interface

- 🎨 **Beautiful colored CLI interface**
- 🤖 **AI-powered spam detection**
- 💬 **Interactive message analysis**
- 📊 **Real-time confidence scores**
- 🔄 **Continuous input mode**
- ⚡ **Fast and accurate predictions**

### 🌐 Web Interface

- 🎨 **Modern dark theme with gradient effects**
- ✨ **Animated background circles**
- 📝 **Interactive textarea with character counter**
- ⚡ **Quick example buttons**
- 📊 **Confidence visualization with progress bars**
- 📱 **Fully responsive design**
- 🎯 **Real-time classification via AJAX**

---

## 🧩 Tech Stack

- **Backend:** Python 🐍, Flask
- **Machine Learning:** scikit-learn (Multinomial Naive Bayes)
- **Data Processing:** pandas, CountVectorizer
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Storage:** Pickle (for model persistence)

---

## 🚀 How to Run

### 1. Activate Virtual Environment

```powershell
cd "c:\my folder\100-days-of-ai"
.\venv\Scripts\Activate.ps1
cd Day-32-Spam-Classifier
```

### 2. Install Dependencies (if needed)

```powershell
pip install pandas scikit-learn flask
```

### 3A. Run the CLI Version

```powershell
python spam_classifier.py
```

### 3B. Run the Web Interface

```powershell
python app.py
```

Then open your browser and go to **http://127.0.0.1:5000**

---

## 🎮 How to Use

### CLI Version:

1. **Launch the program** - You'll see a beautiful banner
2. **Wait for training** - The AI model trains on the dataset
3. **Enter messages** - Type any message you want to check
4. **Get results** - See if it's spam or legitimate with confidence score
5. **Keep testing** - Try as many messages as you want
6. **Exit** - Type 'quit' or 'exit' when done

### Web Version:

1. **Start the Flask server** - Run `python app.py`
2. **Open browser** - Navigate to http://127.0.0.1:5000
3. **Enter message** - Type or paste any message in the textarea
4. **Try examples** - Click quick example buttons for instant testing
5. **View results** - See spam/legitimate classification with confidence bar
6. **Continuous testing** - No page reload needed, instant results!

---

## 📊 Example Usage

### CLI Output:

```
Enter a message to analyze: Congratulations! You won $1000!

────────────────────────────────────────────────────────────
📩 Your Message:
   "Congratulations! You won $1000!"
────────────────────────────────────────────────────────────

🚨 RESULT: SPAM DETECTED! 🚨
   This message appears to be spam/unwanted.
   Confidence: 95.3%
   ⚠️  Be cautious! Do not click links or respond.

────────────────────────────────────────────────────────────
```

### Web Interface:

- Beautiful gradient backgrounds
- Animated confidence bars
- Color-coded results (red for spam, green for legitimate)
- Quick example buttons for testing

---

## 📊 Model Performance

- ✅ **Accuracy: 98.88%** on test set
- 📈 Trained on **5,776 unique messages**
  - 📧 **4,872 Ham** (legitimate) messages
  - 🚨 **904 Spam** messages
- 🧠 Uses **Multinomial Naive Bayes** algorithm
- 🔤 **CountVectorizer** with 5,000 max features
- 💾 **Model persistence** via pickle files (instant loading on restart)

### Classification Report:

```
              precision    recall    f1-score
Ham (Legit)      99%        99%        99%
Spam             97%        96%        96%
Overall       98.88% accuracy
```

## 📁 Datasets Used

The model is trained on **3 combined datasets**:

1. **spam.csv** - 47 messages (custom curated)
2. **SMSSpamCollection** - 5,572 SMS messages (tab-separated)
3. **spam-ham v2.csv** - 5,572 messages (CSV format)

After combining and removing duplicates: **5,776 unique messages**

## 💾 Model Persistence

The trained model is automatically saved as pickle files:

- `spam_classifier_model.pkl` - The trained classifier
- `vectorizer.pkl` - The text vectorizer

**Benefits:**

- ⚡ **Instant startup** on subsequent runs
- 🚫 **No retraining** required unless datasets change
- 💪 **Production-ready** for deployment

To retrain the model, simply delete the `.pkl` files and restart the app.

## 🧪 Try These Examples

**Spam Messages:**

- "You won a free iPhone! Claim now!"
- "URGENT! Your account will be suspended"
- "Get rich quick! Limited time offer"
- "Congratulations! You won $1000! Click here"

**Legitimate Messages:**

- "Let's meet at 6pm for dinner"
- "Can you send me the project report?"
- "Happy birthday! Have a great day"
- "Meeting rescheduled to tomorrow at 3pm"

## 🎨 Interface Features

### CLI:

- 🎨 **Color-coded results** (Red for spam, Green for legitimate)
- 📊 **Confidence percentages** for each prediction
- 🔄 **Continuous analysis mode** - no need to restart
- 🛡️ **Safety warnings** for detected spam
- ✨ **Clean and professional design**

### Web:

- 🎨 **Modern dark theme** (#0f172a background)
- 🌈 **Purple/pink gradients** for visual appeal
- ✨ **Smooth animations** and transitions
- 📊 **Real-time accuracy badge** in header
- 🎯 **4 Feature cards** explaining benefits
- 📱 **Mobile responsive** design

## 📁 Project Structure

```
Day-32-Spam-Classifier/
├── app.py                          # Flask web application
├── spam_classifier.py              # CLI version
├── spam.csv                        # Original dataset
├── SMSSpamCollection              # SMS spam dataset
├── spam-ham v2.csv                # Additional dataset
├── spam_classifier_model.pkl      # Saved ML model
├── vectorizer.pkl                 # Saved vectorizer
├── templates/
│   └── index.html                 # Web interface HTML
├── static/
│   ├── style.css                  # Modern CSS styling
│   └── script.js                  # Frontend JavaScript
└── README.md                      # This file
```

## 🔧 Customization

You can enhance the classifier by:

- Adding more training data to the datasets
- Adjusting the vectorizer parameters in `app.py`
- Trying different ML algorithms (SVM, Random Forest, etc.)
- Adding more features (e.g., message length, special characters)
- Customizing the web interface colors and animations
- Implementing user authentication
- Adding message history tracking
- Creating API endpoints for external use

## 🌐 Web Interface Technical Details

### Backend (`app.py`):

- **Framework:** Flask with debug mode
- **Routes:**
  - `/` - Main page
  - `/classify` - POST endpoint for classification
  - `/stats` - GET model statistics
- **CORS:** Disabled (local use only)
- **Port:** 5000
- **Host:** 0.0.0.0 (accessible on local network)

### Frontend:

- **No dependencies** - Pure HTML/CSS/JS
- **AJAX requests** via Fetch API
- **Font Awesome** icons for visual appeal
- **CSS Grid** for responsive layouts
- **CSS animations** for smooth transitions

## 📚 Learning Outcomes

From this project, you'll learn:

- ✅ **Text classification** with Naive Bayes
- ✅ **Feature extraction** with CountVectorizer
- ✅ **Model persistence** using pickle
- ✅ **Flask web development** basics
- ✅ **RESTful API** design

## 🚀 Future Enhancements

Potential improvements:

- [ ] Add more datasets for better accuracy
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Create a browser extension
- [ ] Add email integration
- [ ] Implement user feedback loop
- [ ] Create mobile app version
- [ ] Add multi-language support
- [ ] Deploy to cloud (Heroku, AWS, Azure)
- [ ] Add analytics dashboard
- [ ] Implement A/B testing

---

## 📝 License

This project is for educational purposes as part of the 100 Days of AI challenge.

---

## 🤝 Contributing

Feel free to fork this project and add your own enhancements!

---

**Built with ❤️ for Day 32 of 100 Days of AI**
