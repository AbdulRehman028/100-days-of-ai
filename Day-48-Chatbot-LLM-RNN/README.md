# Day 48: Chatbot with LLM/RNN 🤖

A conversational AI chatbot powered by DialoGPT, a large language model trained on 147 million Reddit conversations!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green.svg)
![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow.svg)

## 🎯 What This Project Does

This project implements an **end-to-end conversational chatbot** using a Large Language Model (LLM):

- **DialoGPT**: GPT-2 based model fine-tuned for dialogue
- **Context Awareness**: Maintains conversation history for coherent responses
- **Autoregressive Generation**: Generates text token-by-token
- **Real-time Chat**: Interactive web interface with Flask

## 🧠 How It Works

### Seq2Seq Architecture (Simplified)

Traditional seq2seq models use:
```
Encoder → Context Vector → Decoder
```

DialoGPT uses a **decoder-only transformer** (like GPT-2):
```
[History + User Input] → DialoGPT → [Generated Response]
```

### The Generation Process

1. **Tokenization**: User message → Token IDs
2. **Context Building**: Concatenate with conversation history
3. **Generation**: Autoregressive token-by-token generation
4. **Decoding**: Token IDs → Human readable text

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_new_tokens` | 150 | Maximum response length |
| `temperature` | 0.7 | Controls randomness |
| `top_k` | 50 | Limits vocabulary choices |
| `top_p` | 0.95 | Nucleus sampling threshold |
| `no_repeat_ngram` | 3 | Prevents repetition |

## 🚀 Features

- 🤖 **DialoGPT-medium** - 345M parameter conversational model
- 💬 **Context Memory** - Maintains conversation history
- 🔄 **Autoregressive** - Token-by-token generation
- 🎛️ **Controllable** - Temperature, top-k, top-p sampling
- 🌐 **Web Interface** - Modern chat UI
- 📊 **Stats Dashboard** - Model info and message count

## 📦 Installation

1. **Navigate to the project folder:**
   ```bash
   cd Day-48-Chatbot-LLM-RNN
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   
   Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application:**
   ```bash
   python app.py
   ```

6. **Open in browser:**
   ```
   http://localhost:5000
   ```

> ⚠️ **Note**: First run will download the model (~1.4GB for medium). Be patient!

## 🏗️ Project Structure

```
Day-48-Chatbot-LLM-RNN/
├── app.py                # Main Flask app with DialoGPT
├── requirements.txt      # Python dependencies
├── README.md            # Documentation
├── .gitignore           # Git ignore file
├── venv/                # Virtual environment
└── templates/
    └── index.html       # Chat interface
```

## 🔬 DialoGPT Model Variants

| Model | Parameters | Size | Speed | Quality |
|-------|------------|------|-------|---------|
| DialoGPT-small | 117M | ~500MB | ⚡ Fast | Good |
| DialoGPT-medium | 345M | ~1.4GB | 🔄 Balanced | Better |
| DialoGPT-large | 762M | ~3GB | 🐢 Slow | Best |

To change the model, edit `app.py`:
```python
chatbot = ConversationalChatbot(model_name="microsoft/DialoGPT-large")
```

## 🎮 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/chat` | POST | Send message, get response |
| `/clear` | POST | Clear conversation history |
| `/history` | GET | Get conversation history |
| `/stats` | GET | Get model statistics |

## 🧪 Example Usage

### Chat Request
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! How are you?"}'
```

### Response
```json
{
  "success": true,
  "response": "I'm doing great, thanks for asking! How about you?",
  "confidence": 0.65,
  "model": "DialoGPT-medium"
}
```

## 📊 Comparison: RNN vs Transformer

| Feature | RNN/LSTM | Transformer (DialoGPT) |
|---------|----------|------------------------|
| Architecture | Recurrent | Self-Attention |
| Parallelization | ❌ Sequential | ✅ Parallel |
| Long-range deps | 😐 Limited | ✅ Excellent |
| Training time | 🐢 Slow | ⚡ Fast |
| Context window | ~100 tokens | 1024 tokens |

### Why DialoGPT over RNN?

1. **Better Context**: Attention mechanism captures long-range dependencies
2. **Pre-trained**: 147M Reddit conversations = rich conversational knowledge
3. **Quality**: More coherent, contextual responses
4. **No Training**: Ready to use out-of-the-box

## 🎓 Learning Outcomes

By building this project, you'll learn:

1. **Transformers** - How decoder-only models generate text
2. **Autoregressive Generation** - Token-by-token prediction
3. **Conversation Context** - Maintaining chat history
4. **Sampling Strategies** - Temperature, top-k, top-p
5. **Hugging Face** - Using pre-trained models

## 🔧 Customization

### Adjust Response Style

```python
# More creative responses
output_ids = self.model.generate(
    bot_input_ids,
    temperature=1.0,      # Higher = more random
    top_p=0.9,           # Nucleus sampling
)

# More focused responses  
output_ids = self.model.generate(
    bot_input_ids,
    temperature=0.3,      # Lower = more deterministic
    top_k=10,            # Fewer choices
)
```

### Change Model
```python
# Smaller, faster model
chatbot = ConversationalChatbot("microsoft/DialoGPT-small")

# Larger, better model (needs more RAM)
chatbot = ConversationalChatbot("microsoft/DialoGPT-large")
```

## 🔗 Related Projects

- **Day 46**: Rule-Based Chatbot (Pattern Matching)
- **Day 47**: FAQ Chatbot with Embeddings
- **Day 35**: GPT-2 Text Generation
- **Day 33**: LSTM Text Generator

## 📈 Day 46 → 47 → 48 Evolution

| Day | Approach | How it Works |
|-----|----------|--------------|
| 46 | Rule-Based | Regex pattern matching |
| 47 | Embeddings | Semantic similarity search |
| **48** | **LLM** | **Autoregressive generation** |

## 📝 License

This project is part of the 100 Days of AI challenge.

---