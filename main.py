import os
from scripts.preprocess import preprocess_data
from scripts.train import train_model
from scripts.evaluate import evaluate_model
import mlflow

# Define paths
DATA_DIR = "./data"
MODELS_DIR = "./models"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

def main():
    """
    Main function to orchestrate the pipeline: Preprocessing, Training, and Evaluation.
    """
    # Ensure necessary directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Set MLflow tracking URI and experiment name
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Sentiment Analysis")

    # Step 1: Preprocess data
    print("Starting data preprocessing...")
    train_data_path = os.path.join(DATA_DIR, "training.1600000.processed.noemoticon.csv")
    test_data_path = os.path.join(DATA_DIR, "testdata.manual.2009.06.14.csv")
    processed_data = preprocess_data(train_data_path, test_data_path)
    print("Data preprocessing complete.")

    # Step 2: Train the model
    print("Starting model training...")
    model_artifacts = train_model(processed_data, MODELS_DIR)
    print("Model training complete.")

    # Step 3: Evaluate the model
    print("Starting model evaluation...")
    evaluate_model(processed_data, model_artifacts)
    print("Model evaluation complete.")

    print("Pipeline execution finished successfully.")

if _name_ == "_main_":
    main()