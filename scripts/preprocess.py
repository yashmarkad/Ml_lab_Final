import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

def preprocess_data(train_path, test_path):
    """
    Preprocesses training and test datasets:
    - Cleans and tokenizes tweets.
    - Removes stopwords.
    - Applies lemmatization.

    Args:
        train_path (str): Path to the training dataset CSV file.
        test_path (str): Path to the test dataset CSV file.

    Returns:
        pd.DataFrame: Combined and preprocessed DataFrame.
    """
    # Load data
    train = pd.read_csv(train_path, encoding="latin1", header=None)
    test = pd.read_csv(test_path, encoding="latin1", header=None)

    # Define column names
    train.columns = ["Polarity", "id", "date", "query", "user", "tweet"]
    test.columns = ["Polarity", "id", "date", "query", "user", "tweet"]

    # Select relevant columns
    train = train[["Polarity", "tweet"]]
    test = test[["Polarity", "tweet"]]

    # Standardize polarity labels
    train["Polarity"] = train["Polarity"].replace(4, 1).replace(0, -1)
    test["Polarity"] = test["Polarity"].replace(4, 1).replace(0, -1).replace(2, 0)

    # Combine datasets
    data = pd.concat([train, test], axis=0)

    # Preprocessing helper functions
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def clean_text(text):
        """
        Cleans text by removing unwanted characters and patterns.
        """
        text = re.sub(r"@[\w]+", "", text)  # Remove mentions
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"[^A-Za-z\s]", "", text)  # Remove non-alphanumeric characters
        return text.lower().strip()

    def tokenize_and_lemmatize(text):
        """
        Tokenizes text, removes stopwords, and applies lemmatization.
        """
        tokens = text.split()
        cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(cleaned_tokens)

    # Apply cleaning and tokenization
    data["cleaned_tweet"] = data["tweet"].astype(str).apply(clean_text)
    data["cleaned_tweet"] = data["cleaned_tweet"].apply(tokenize_and_lemmatize)

    # Remove empty tweets after preprocessing
    data = data[data["cleaned_tweet"].str.strip() != ""]

    return data