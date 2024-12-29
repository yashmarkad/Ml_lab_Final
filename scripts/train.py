import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_model(data, model_dir):
    """
    Trains a logistic regression model for sentiment analysis, logs with MLflow,
    and saves the trained model and vectorizer.

    Args:
        data (pd.DataFrame): Preprocessed DataFrame with 'cleaned_tweet' and 'Polarity'.
        model_dir (str): Directory to save the trained model and vectorizer.

    Returns:
        dict: Paths to the saved model and vectorizer, along with accuracy.
    """
    # Split the data
    X = data["cleaned_tweet"]
    y = data["Polarity"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Vectorize text using TF-IDF
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Start an MLflow run
    with mlflow.start_run():
        # Train a Logistic Regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_tfidf, y_train)

        # Make predictions and calculate accuracy
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters, metrics, and model with MLflow
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "sentiment_model")

        # Save the model and vectorizer locally
        model_path = f"{model_dir}/sentiment_model.pkl"
        tfidf_path = f"{model_dir}/tfidf_model.pkl"
        joblib.dump(model, model_path)
        joblib.dump(tfidf, tfidf_path)

        # Log artifacts
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(tfidf_path)

        print(f"Model trained with accuracy: {accuracy:.4f}")
        return {"model_path": model_path, "tfidf_path": tfidf_path, "accuracy":accuracy}