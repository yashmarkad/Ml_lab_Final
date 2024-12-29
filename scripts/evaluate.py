import joblib
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def evaluate_model(data, artifacts):
    """
    Evaluates the trained model on the provided dataset and generates metrics.

    Args:
        data (pd.DataFrame): Preprocessed DataFrame with 'cleaned_tweet' and 'Polarity'.
        artifacts (dict): Paths to the saved model and vectorizer.

    Returns:
        dict: Evaluation results, including accuracy and classification report.
    """
    # Load the trained model and vectorizer
    model = joblib.load(artifacts["model_path"])
    tfidf = joblib.load(artifacts["tfidf_path"])

    # Prepare data for evaluation
    X = data["cleaned_tweet"]
    y = data["Polarity"]
    X_tfidf = tfidf.transform(X)

    # Predict sentiments
    y_pred = model.predict(X_tfidf)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=["Negative", "Neutral", "Positive"])

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(report)

    # Save the classification report to a file
    report_path = "./models/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    return {
        "accuracy": accuracy,
        "report": report,
        "report_path": report_path,
    }