# �️ Day 32 — AI Spam Classifier (Interactive)

### 🎯 Goal
Build an **interactive Spam/Ham Classifier** with a beautiful CLI interface using Naive Bayes and text vectorization.

---

## ✨ Features
- 🎨 **Beautiful colored CLI interface**
- 🤖 **AI-powered spam detection**
- 💬 **Interactive message analysis**
- 📊 **Real-time confidence scores**
- 🔄 **Continuous input mode**
- ⚡ **Fast and accurate predictions**

---

## 🧩 Tech Stack
- Python 🐍  
- scikit-learn (Machine Learning)
- pandas (Data handling)
- CountVectorizer (Text processing)
- Naive Bayes Classifier

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
pip install pandas scikit-learn matplotlib seaborn
```

### 3. Run the Spam Classifier
```powershell
python spam_classifier.py
```

---

## 🎮 How to Use

1. **Launch the program** - You'll see a beautiful banner
2. **Wait for training** - The AI model trains on the dataset
3. **Enter messages** - Type any message you want to check
4. **Get results** - See if it's spam or legitimate with confidence score
5. **Keep testing** - Try as many messages as you want
6. **Exit** - Type 'quit' or 'exit' when done

---

## � Example Usage

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

---

## 📊 Model Performance
- ✅ **Accuracy: 100%** on test set
- 📈 Trained on **47 messages** (24 ham, 23 spam)
- 🧠 Uses **Multinomial Naive Bayes** algorithm
- 🔤 **CountVectorizer** for text feature extraction

---

## 🧪 Try These Examples

**Spam Messages:**
- "You won a free iPhone! Claim now!"
- "URGENT! Your account will be suspended"
- "Get rich quick! Limited time offer"

**Legitimate Messages:**
- "Let's meet at 6pm for dinner"
- "Can you send me the report?"
- "Happy birthday! Have a great day"

---

## 🎨 Interface Features

- 🎨 **Color-coded results** (Red for spam, Green for legitimate)
- 📊 **Confidence percentages** for each prediction
- 🔄 **Continuous analysis mode** - no need to restart
- 🛡️ **Safety warnings** for detected spam
- ✨ **Clean and professional design**

---

## 📝 Dataset
- Custom curated dataset in `spam.csv`
- Contains examples of spam and legitimate messages
- Can be expanded with more examples

For a larger dataset, visit:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

---

## 🔧 Customization

You can enhance the classifier by:
- Adding more training data to `spam.csv`
- Adjusting the vectorizer parameters
- Trying different ML algorithms
- Adding more features (e.g., message length, special characters)