"""
Maillim - End-to-End Email Assistant
Day 55: 100 Days of AI

An intelligent email assistant that uses LLMs to:
- Triage emails into Urgent, Routine, or Spam
- Generate context-aware draft responses
- Analyze and match tone/sentiment
"""

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from imap_tools import MailBox, AND
from dataclasses import dataclass
from typing import Optional, List
import torch

# =====================
# Data Classes
# =====================
@dataclass
class Email:
    """Represents an email message"""
    uid: str
    sender: str
    subject: str
    body: str
    date: str
    
@dataclass
class TriagedEmail:
    """Email with triage classification and analysis"""
    email: Email
    category: str  # Urgent, Routine, Spam
    confidence: float
    sentiment: str
    sentiment_score: float
    draft_response: Optional[str] = None


# =====================
# Maillim Core Engine
# =====================
class MaillimEngine:
    """Core AI engine for email processing"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_models()
        
    @st.cache_resource
    def _init_models(_self):
        """Initialize all ML pipelines with caching"""
        # Classification pipeline for triage
        _self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if _self.device == "cuda" else -1
        )
        
        # Sentiment analysis pipeline
        _self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if _self.device == "cuda" else -1
        )
        
        # Text generation for drafting responses (Flan-T5)
        _self.response_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        _self.response_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        if _self.device == "cuda":
            _self.response_model = _self.response_model.to(_self.device)
            
    def triage_email(self, email_text: str) -> tuple[str, float]:
        """
        Classify email into Urgent, Routine, or Spam
        
        Args:
            email_text: Combined subject and body text
            
        Returns:
            Tuple of (category, confidence_score)
        """
        candidate_labels = ["urgent", "routine", "spam"]
        
        # Truncate text to avoid token limits
        truncated_text = email_text[:1000]
        
        result = self.classifier(
            truncated_text,
            candidate_labels,
            hypothesis_template="This email is {}."
        )
        
        category = result["labels"][0].capitalize()
        confidence = result["scores"][0]
        
        return category, confidence
    
    def analyze_sentiment(self, text: str) -> tuple[str, float]:
        """
        Analyze the sentiment/tone of the email
        
        Args:
            text: Email text to analyze
            
        Returns:
            Tuple of (sentiment_label, confidence_score)
        """
        truncated_text = text[:512]
        result = self.sentiment_analyzer(truncated_text)[0]
        
        return result["label"], result["score"]
    
    def generate_response(self, email: Email, sentiment: str) -> str:
        """
        Generate a context-aware draft response using Flan-T5
        
        Args:
            email: The original email object
            sentiment: Detected sentiment of original email
            
        Returns:
            Generated draft response text
        """
        # Craft prompt based on sentiment and context
        tone_instruction = self._get_tone_instruction(sentiment)
        
        prompt = f"""Write a professional email reply to the following email.
{tone_instruction}

Original Email:
Subject: {email.subject}
From: {email.sender}
Content: {email.body[:500]}

Draft a polite and helpful response:"""

        inputs = self.response_tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.response_model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            early_stopping=True
        )
        
        response = self.response_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def _get_tone_instruction(self, sentiment: str) -> str:
        """Get tone instruction based on detected sentiment"""
        if sentiment == "NEGATIVE":
            return "The original email has a concerned or negative tone. Respond with empathy and reassurance."
        elif sentiment == "POSITIVE":
            return "The original email has a positive tone. Match the enthusiasm in your response."
        else:
            return "Maintain a professional and neutral tone."
    
    def process_email(self, email: Email) -> TriagedEmail:
        """
        Full processing pipeline for an email
        
        Args:
            email: Email object to process
            
        Returns:
            TriagedEmail with all analysis results
        """
        combined_text = f"{email.subject} {email.body}"
        
        # Step 1: Triage classification
        category, confidence = self.triage_email(combined_text)
        
        # Step 2: Sentiment analysis
        sentiment, sentiment_score = self.analyze_sentiment(combined_text)
        
        # Step 3: Generate draft response (skip for spam)
        draft_response = None
        if category != "Spam":
            draft_response = self.generate_response(email, sentiment)
        
        return TriagedEmail(
            email=email,
            category=category,
            confidence=confidence,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            draft_response=draft_response
        )


# =====================
# Email Fetcher (IMAP)
# =====================
class EmailFetcher:
    """Fetches emails from IMAP server"""
    
    def __init__(self, server: str, email: str, password: str):
        self.server = server
        self.email = email
        self.password = password
        
    def fetch_emails(self, folder: str = "INBOX", limit: int = 10) -> List[Email]:
        """
        Fetch recent emails from the mailbox
        
        Args:
            folder: Mailbox folder to fetch from
            limit: Maximum number of emails to fetch
            
        Returns:
            List of Email objects
        """
        emails = []
        
        with MailBox(self.server).login(self.email, self.password) as mailbox:
            mailbox.folder.set(folder)
            
            for msg in mailbox.fetch(limit=limit, reverse=True):
                email = Email(
                    uid=msg.uid,
                    sender=msg.from_,
                    subject=msg.subject or "(No Subject)",
                    body=msg.text or msg.html or "(No Content)",
                    date=str(msg.date)
                )
                emails.append(email)
                
        return emails


# =====================
# Streamlit UI
# =====================
def main():
    st.set_page_config(
        page_title="Maillim - Email Assistant",
        page_icon="📧",
        layout="wide"
    )
    
    st.title("📧 Maillim - AI Email Assistant")
    st.markdown("*Intelligent email triage, sentiment analysis, and response generation*")
    
    # Initialize engine
    @st.cache_resource
    def get_engine():
        return MaillimEngine()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        mode = st.radio(
            "Input Mode",
            ["Manual Entry", "IMAP Connection"],
            help="Choose how to input emails for processing"
        )
        
        if mode == "IMAP Connection":
            st.subheader("📬 IMAP Settings")
            imap_server = st.text_input("IMAP Server", placeholder="imap.gmail.com")
            email_address = st.text_input("Email Address", placeholder="you@example.com")
            email_password = st.text_input("App Password", type="password")
            email_limit = st.slider("Emails to Fetch", 1, 20, 5)
            
            fetch_button = st.button("🔄 Fetch Emails", type="primary")
        
        st.divider()
        st.markdown("""
        **Triage Categories:**
        - 🔴 **Urgent**: Requires immediate attention
        - 🟡 **Routine**: Standard correspondence
        - ⚫ **Spam**: Unwanted/promotional content
        """)
    
    # Main content area
    engine = get_engine()
    
    if mode == "Manual Entry":
        st.header("✍️ Manual Email Entry")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sender = st.text_input("From", placeholder="sender@example.com")
            subject = st.text_input("Subject", placeholder="Enter email subject")
            
        with col2:
            email_date = st.text_input("Date", value="Today")
            
        body = st.text_area(
            "Email Body",
            placeholder="Paste or type the email content here...",
            height=200
        )
        
        if st.button("🚀 Process Email", type="primary"):
            if body.strip():
                with st.spinner("Processing email with AI..."):
                    email = Email(
                        uid="manual-001",
                        sender=sender or "unknown@example.com",
                        subject=subject or "(No Subject)",
                        body=body,
                        date=email_date
                    )
                    
                    result = engine.process_email(email)
                    display_results(result)
            else:
                st.warning("Please enter email content to process.")
                
    else:  # IMAP Connection mode
        st.header("📬 Email Inbox")
        
        if 'fetched_emails' not in st.session_state:
            st.session_state.fetched_emails = []
            st.session_state.processed_results = []
            
        if mode == "IMAP Connection" and 'fetch_button' in dir() and fetch_button:
            if all([imap_server, email_address, email_password]):
                try:
                    with st.spinner("Connecting to mailbox..."):
                        fetcher = EmailFetcher(imap_server, email_address, email_password)
                        st.session_state.fetched_emails = fetcher.fetch_emails(limit=email_limit)
                        st.session_state.processed_results = []
                        st.success(f"Fetched {len(st.session_state.fetched_emails)} emails!")
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
            else:
                st.warning("Please fill in all IMAP connection details.")
        
        # Display fetched emails
        if st.session_state.fetched_emails:
            st.subheader(f"📨 {len(st.session_state.fetched_emails)} Emails Found")
            
            if st.button("🤖 Process All Emails"):
                progress = st.progress(0)
                st.session_state.processed_results = []
                
                for i, email in enumerate(st.session_state.fetched_emails):
                    with st.spinner(f"Processing email {i+1}/{len(st.session_state.fetched_emails)}..."):
                        result = engine.process_email(email)
                        st.session_state.processed_results.append(result)
                    progress.progress((i + 1) / len(st.session_state.fetched_emails))
                    
                st.success("All emails processed!")
            
            # Display results
            if st.session_state.processed_results:
                display_batch_results(st.session_state.processed_results)
            else:
                for email in st.session_state.fetched_emails:
                    with st.expander(f"📧 {email.subject[:50]}..."):
                        st.write(f"**From:** {email.sender}")
                        st.write(f"**Date:** {email.date}")
                        st.write(f"**Preview:** {email.body[:200]}...")
        else:
            st.info("Configure IMAP settings and click 'Fetch Emails' to get started.")
    
    # Demo section
    st.divider()
    with st.expander("🎯 Try Demo Emails"):
        demo_emails = [
            {
                "sender": "boss@company.com",
                "subject": "URGENT: Client meeting in 1 hour",
                "body": "Hi, we have an emergency client meeting in 1 hour. Please review the attached proposal and prepare your notes. This is critical for the deal closure. Let me know if you can make it."
            },
            {
                "sender": "newsletter@store.com",
                "subject": "50% OFF - Limited Time Offer!!!",
                "body": "AMAZING DEALS! Click here to save BIG on our latest products. This offer expires soon! Don't miss out on these incredible savings. Unsubscribe link at bottom."
            },
            {
                "sender": "colleague@company.com",
                "subject": "Weekly report submission",
                "body": "Hi, just a reminder to submit your weekly progress report by Friday. Please include the metrics we discussed in our last meeting. Thanks!"
            }
        ]
        
        for i, demo in enumerate(demo_emails):
            if st.button(f"Process Demo {i+1}: {demo['subject'][:30]}...", key=f"demo_{i}"):
                with st.spinner("Processing..."):
                    email = Email(
                        uid=f"demo-{i}",
                        sender=demo["sender"],
                        subject=demo["subject"],
                        body=demo["body"],
                        date="Today"
                    )
                    result = engine.process_email(email)
                    display_results(result)


def display_results(result: TriagedEmail):
    """Display processing results for a single email"""
    
    # Category badge
    category_colors = {
        "Urgent": "🔴",
        "Routine": "🟡", 
        "Spam": "⚫"
    }
    
    sentiment_icons = {
        "POSITIVE": "😊",
        "NEGATIVE": "😟",
        "NEUTRAL": "😐"
    }
    
    st.subheader("📊 Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Category",
            f"{category_colors.get(result.category, '⚪')} {result.category}",
            f"{result.confidence:.1%} confidence"
        )
        
    with col2:
        st.metric(
            "Sentiment",
            f"{sentiment_icons.get(result.sentiment, '😐')} {result.sentiment}",
            f"{result.sentiment_score:.1%} confidence"
        )
        
    with col3:
        priority = "High" if result.category == "Urgent" else ("Low" if result.category == "Spam" else "Normal")
        st.metric("Priority", priority)
    
    # Original email summary
    with st.expander("📨 Original Email", expanded=True):
        st.write(f"**From:** {result.email.sender}")
        st.write(f"**Subject:** {result.email.subject}")
        st.write(f"**Content:** {result.email.body[:500]}...")
    
    # Draft response
    if result.draft_response:
        st.subheader("✉️ AI-Generated Draft Response")
        st.info(result.draft_response)
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("📋 Copy to Clipboard", key=f"copy_{result.email.uid}")
        with col2:
            st.button("✏️ Edit Draft", key=f"edit_{result.email.uid}")
    else:
        st.warning("No draft generated (email classified as Spam)")


def display_batch_results(results: List[TriagedEmail]):
    """Display results for batch processed emails"""
    
    # Summary statistics
    categories = {"Urgent": 0, "Routine": 0, "Spam": 0}
    for r in results:
        categories[r.category] = categories.get(r.category, 0) + 1
    
    st.subheader("📊 Batch Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔴 Urgent", categories["Urgent"])
    with col2:
        st.metric("🟡 Routine", categories["Routine"])
    with col3:
        st.metric("⚫ Spam", categories["Spam"])
    
    st.divider()
    
    # Sort by priority (Urgent first, then Routine, then Spam)
    priority_order = {"Urgent": 0, "Routine": 1, "Spam": 2}
    sorted_results = sorted(results, key=lambda x: priority_order.get(x.category, 3))
    
    for result in sorted_results:
        category_colors = {"Urgent": "🔴", "Routine": "🟡", "Spam": "⚫"}
        icon = category_colors.get(result.category, "⚪")
        
        with st.expander(f"{icon} [{result.category}] {result.email.subject[:40]}..."):
            display_results(result)


if __name__ == "__main__":
    main()
