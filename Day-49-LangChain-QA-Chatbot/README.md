# Day 49: LangChain Q&A Chatbot 🔗

A powerful Q&A chatbot built with LangChain using ChatModel architecture and Hugging Face's Mistral-7B-Instruct!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/🦜🔗-LangChain-green.svg)
![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)

## 🎯 What This Project Does

This project implements a **Q&A chatbot** using LangChain's ChatModel interface:

- **LangChain**: Framework for building LLM applications
- **ChatModel**: Proper chat interface with system/human/AI messages
- **Mistral-7B-Instruct**: Powerful instruction-tuned LLM
- **LCEL**: LangChain Expression Language for clean chains

## 🔗 LangChain Chain Architecture

```
User Input
    │
    ▼
┌─────────────────────────┐
│   ChatPromptTemplate    │  ← System prompt + History + User message
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│    ChatHuggingFace      │  ← Mistral-7B-Instruct via API
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│    StrOutputParser      │  ← Extract string response
└─────────────────────────┘
    │
    ▼
Response
```

## 🚀 Features

- 🔗 **LangChain Integration** - Modern LCEL chain composition
- 💬 **ChatModel Interface** - Proper chat message handling
- 🧠 **Mistral-7B-Instruct** - Powerful instruction-following LLM
- 📜 **Conversation Memory** - Maintains context across messages
- 🎨 **Modern Dark UI** - Beautiful, responsive chat interface
- 🔑 **Token-based Auth** - Secure HuggingFace API integration

## 📦 Installation

1. **Navigate to the project folder:**
   ```bash
   cd Day-49-LangChain-QA-Chatbot
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

## 🔑 Getting Your HuggingFace Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name and select "Read" access
4. Copy the token
5. Paste it in the web interface

**Or set as environment variable:**
```bash
export HF_TOKEN=your_token_here
```

## 🏗️ Project Structure

```
Day-49-LangChain-QA-Chatbot/
├── app.py                # Main Flask app with LangChain
├── requirements.txt      # Python dependencies
├── README.md            # Documentation
├── .gitignore           # Git ignore file
└── templates/
    └── index.html       # Modern chat UI
```

## 🧠 Key LangChain Concepts

### 1. ChatPromptTemplate
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant..."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
```

### 2. ChatHuggingFace (ChatModel)
```python
chat_model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(...),
    verbose=True
)
```

### 3. LCEL Chain Composition
```python
chain = prompt | chat_model | StrOutputParser()
response = chain.invoke({"input": question, "history": history})
```

## 🎮 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/initialize` | POST | Initialize with HF token |
| `/ask` | POST | Send question, get answer |
| `/clear` | POST | Clear conversation history |
| `/history` | GET | Get conversation history |
| `/status` | GET | Check if chatbot is ready |

## 🔬 Why LangChain?

| Feature | Raw API | LangChain |
|---------|---------|-----------|
| Prompt Management | Manual | ✅ Templates |
| Chat History | Manual | ✅ Built-in |
| Output Parsing | Manual | ✅ Parsers |
| Chain Composition | Complex | ✅ LCEL (|) |
| Model Switching | Rewrite | ✅ Swap |

## 📊 LLM vs ChatModel

| Type | Messages | Use Case |
|------|----------|----------|
| **LLM** | Single text input | Completion |
| **ChatModel** | System/Human/AI messages | Conversation |

This project uses **ChatModel** for proper conversational experience!

## 🆚 Day 46 → 47 → 48 → 49 Evolution

| Day | Approach | Intelligence |
|-----|----------|--------------|
| 46 | Rule-Based | Pattern matching |
| 47 | Embeddings | Semantic search |
| 48 | DialoGPT | Casual chat LLM |
| **49** | **LangChain** | **Q&A with ChatModel** |

## 🎓 Learning Outcomes

By building this project, you'll learn:

1. **LangChain Basics** - Chains, prompts, models
2. **ChatModel vs LLM** - Proper chat interfaces
3. **LCEL** - LangChain Expression Language
4. **Prompt Engineering** - System prompts for behavior
5. **HuggingFace Hub** - Using hosted inference API

## 🔧 Customization

### Change the Model
```python
self.model_name = "meta-llama/Llama-2-7b-chat-hf"
# or
self.model_name = "google/flan-t5-xxl"
```

### Adjust Generation Parameters
```python
self.llm = HuggingFaceEndpoint(
    max_new_tokens=1024,    # Longer responses
    temperature=0.9,        # More creative
    top_p=0.85,            # Nucleus sampling
)
```

### Custom System Prompt
```python
self.prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Python expert. Only answer coding questions."),
    ...
])
```

## 📚 Resources

- [LangChain Docs](https://python.langchain.com/docs/)
- [Hugging Face Hub](https://huggingface.co/docs/hub/)
- [Mistral AI](https://mistral.ai/)
- [LCEL Guide](https://python.langchain.com/docs/expression_language/)

## 📝 License

This project is part of the 100 Days of AI challenge.

---

**Day 49 of 100** - Building AI, one day at a time! 🚀
