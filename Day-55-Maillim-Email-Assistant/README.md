# Day 55: Maillim - AI Email Assistant 📧

An end-to-end intelligent email assistant that uses Large Language Models (LLMs) to triage incoming emails, analyze sentiment, and generate context-aware draft responses.

## Features

### 🎯 Email Triage
Automatically classifies emails into three categories using zero-shot classification:
- **🔴 Urgent**: Requires immediate attention
- **🟡 Routine**: Standard correspondence
- **⚫ Spam**: Unwanted/promotional content

### 💬 Sentiment Analysis
Analyzes the emotional tone of incoming emails to:
- Detect positive, negative, or neutral sentiment
- Inform response tone matching
- Provide confidence scores

### ✉️ AI Response Generation
Generates context-aware draft responses using Flan-T5:
- Adapts tone based on original email sentiment
- Maintains professional language
- Provides one-click copy functionality

### 📬 IMAP Integration
Connect directly to your email inbox:
- Supports Gmail, Outlook, Yahoo, and other IMAP servers
- Batch processing of multiple emails
- Priority-sorted results view

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ML Framework | Hugging Face Transformers |
| Classification | BART (facebook/bart-large-mnli) |
| Sentiment | DistilBERT fine-tuned on SST-2 |
| Generation | Flan-T5 (google/flan-t5-base) |
| Email Access | imap_tools |
| UI | Streamlit |

## Installation

1. **Clone or navigate to the project directory:**
```bash
cd Day-55-Maillim-Email-Assistant
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
streamlit run maillim_app.py
```

### Manual Mode
1. Select "Manual Entry" mode in the sidebar
2. Enter email details (sender, subject, body)
3. Click "Process Email"
4. View triage results and generated response

### IMAP Mode
1. Select "IMAP Connection" mode
2. Enter your IMAP server details:
   - Server: `imap.gmail.com` (for Gmail)
   - Email: Your email address
   - Password: App-specific password (see below)
3. Click "Fetch Emails"
4. Process individual or all emails

### Gmail Setup (App Password)
1. Enable 2-Factor Authentication on your Google account
2. Go to Google Account → Security → App Passwords
3. Generate a new app password for "Mail"
4. Use this password in the application

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Maillim Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌────────────┐    ┌──────────────────────┐ │
│  │  Email  │───▶│   Triage   │───▶│ Urgent/Routine/Spam │ │
│  │  Input  │    │  (BART)    │    │   Classification     │ │
│  └─────────┘    └────────────┘    └──────────────────────┘ │
│       │                                                     │
│       ▼                                                     │
│  ┌────────────────┐    ┌────────────────────────────────┐  │
│  │   Sentiment    │───▶│  Positive/Negative/Neutral    │  │
│  │  (DistilBERT)  │    │      Tone Detection           │  │
│  └────────────────┘    └────────────────────────────────┘  │
│       │                                                     │
│       ▼                                                     │
│  ┌────────────────┐    ┌────────────────────────────────┐  │
│  │   Response     │───▶│  Context-Aware Draft Reply    │  │
│  │   (Flan-T5)    │    │   (Tone-Matched)              │  │
│  └────────────────┘    └────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Models Used

### 1. Zero-Shot Classification (Triage)
- **Model**: `facebook/bart-large-mnli`
- **Purpose**: Classify emails without task-specific training
- **Labels**: Urgent, Routine, Spam

### 2. Sentiment Analysis
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Purpose**: Detect emotional tone
- **Output**: POSITIVE, NEGATIVE with confidence score

### 3. Text Generation (Response Drafting)
- **Model**: `google/flan-t5-base`
- **Purpose**: Generate coherent, context-aware replies
- **Features**: Instruction-following, tone matching

## Example Output

```
Input Email:
From: boss@company.com
Subject: URGENT: Client meeting in 1 hour
Body: We have an emergency client meeting...

Analysis Results:
├── Category: 🔴 Urgent (94.2% confidence)
├── Sentiment: 😟 NEGATIVE (78.5% confidence)
└── Priority: High

Generated Draft:
"Thank you for the heads up. I'm reviewing the proposal now
and will be prepared for the meeting. I'll have my notes
ready. Please let me know if there's anything specific you'd
like me to focus on."
```

## Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| Input Mode | Manual Entry or IMAP Connection | Manual |
| Email Limit | Number of emails to fetch (IMAP) | 5 |
| IMAP Server | Email server address | - |

## Limitations

- Models run locally (requires ~4GB RAM minimum)
- First run downloads models (~2GB)
- GPU recommended for faster processing
- Response quality depends on email clarity

## Future Enhancements

- [ ] Multi-language support
- [ ] Custom classification categories
- [ ] Email sending capability
- [ ] Calendar integration for scheduling
- [ ] Contact-based personalization
- [ ] Fine-tuned models for better accuracy

## License

MIT License - Part of the 100 Days of AI Challenge

## Author

100 Days of AI - Day 55
