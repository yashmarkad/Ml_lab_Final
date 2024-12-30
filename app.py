import streamlit as st
import joblib
import pandas as pd
import nltk
nltk.data.path.append('/path/to/nltk_data') 
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse

# Load Models
tfidf = joblib.load("model/tidf_model.pkl")
length_scaler = joblib.load("model/length_model.pkl")
sentiment_model = joblib.load("model/Sentiment_model2.pkl")

# Initialize Lemmatizer
lem = WordNetLemmatizer()

def clean_text(text):
    pat1 = r'@[^ ]+'
    pat2 = r'https?://[A-Za-z0-9./]+'
    pat3 = r'\'s'
    pat4 = r'\#\w+'
    pat5 = r'&amp '
    pat6 = r'[^A-Za-z\s]'
    combined_pat = r'|'.join((pat1, pat2, pat3, pat4, pat5, pat6))
    text = re.sub(combined_pat, "", text).lower()
    return text.strip()

def tokenize_lem(sentence):
    outlist = []
    token = sentence.split()
    for tok in token:
        outlist.append(lem.lemmatize(tok))
    return " ".join(outlist)

def preprocess_input(text):
    # Clean and lemmatize text
    cleaned_text = clean_text(text)
    lemmatized_text = tokenize_lem(cleaned_text)
    
    # Transform using TFIDF
    text_vector = tfidf.transform([lemmatized_text])
    
    # Calculate length and scale it
    length = len(lemmatized_text)
    length_scaled = length_scaler.transform([[length]])
    
    # Combine features
    combined_features = scipy.sparse.hstack([text_vector, length_scaled], format="csr")
    return combined_features

# Streamlit App
st.title("Sentiment Analysis Application")
st.write("This app uses a sentiment analysis model to predict the sentiment of a tweet.")

# User Input
user_input = st.text_area("Enter your tweet:", "")

if st.button("Predict Sentiment"):
    if user_input:
        features = preprocess_input(user_input)
        prediction = sentiment_model.predict(features)[0]
        sentiment = "Positive" if prediction == 1 else "Negative" if prediction == -1 else "Neutral"
        st.write(f"The predicted sentiment is: *{sentiment}*")
    else:
        st.write("Please enter a tweet to analyze.")
