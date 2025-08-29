import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import json
from io import BytesIO
import re
import string

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

st.title("Text Preprocessor for RAG")

# Text input or file upload
text_input = st.text_area("Enter text")
uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])

# Options
remove_stopwords_option = st.checkbox("Remove stopwords", value=True)
lemmatize_option = st.checkbox("Lemmatize words", value=True)
generate_embeddings_option = st.checkbox("Generate embeddings", value=True)
remove_emojis_option = st.checkbox("Remove emojis", value=True)
remove_punctuation_option = st.checkbox("Remove punctuation", value=True)
remove_hashtags_option = st.checkbox("Remove hashtags", value=True)

def preprocess_text(text):
    # 1. Lowercase everything
    text = text.lower()

    # 2. Remove emojis
    if remove_emojis_option:
        # A simple regex pattern to remove most emojis
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
    
    # 3. Remove hashtags
    if remove_hashtags_option:
        text = re.sub(r'#\w+', '', text)

    # 4. Remove punctuation
    if remove_punctuation_option:
        text = text.translate(str.maketrans('', '', string.punctuation))

    # 5. Tokenize
    tokens = text.split()

    # 6. Remove stopwords
    if remove_stopwords_option:
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]

    # 7. Lemmatize words
    if lemmatize_option:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)

if text_input or uploaded_file:
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
    else:
        text = text_input
    
    if st.button("Preprocess Text"):
        cleaned_text = preprocess_text(text)
        st.write("Cleaned Text:")
        st.text(cleaned_text)
        
        # Word cloud
        if cleaned_text:
            wordcloud = WordCloud(width=800, height=400).generate(cleaned_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        
        # Embeddings
        if generate_embeddings_option:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(cleaned_text).tolist()
            st.write("Embedding Preview (first 5 values):")
            st.write(embedding[:5])
            
            # Download embeddings
            embedding_json = json.dumps({"text": cleaned_text, "embedding": embedding}, indent=2)
            st.download_button(
                label="Download Embeddings (JSON)",
                data=embedding_json,
                file_name="text_embedding.json",
                mime="application/json"
            )
        
        # Download cleaned text
        st.download_button(
            label="Download Cleaned Text",
            data=cleaned_text,
            file_name="cleaned_text.txt",
            mime="text/plain"
        )