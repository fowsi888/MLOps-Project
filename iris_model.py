"""
Iris Classification with MLflow

This script trains a decision tree model on the Iris dataset,
saves it using MLflow, loads it back, and evaluates its performance.
"""
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# Constants
EXPERIMENT_NAME = "iris_classification_new"
MODEL_NAME = "iris_decision_tree"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature and class names for the Iris dataset
FEATURE_NAMES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm"
]

CLASS_NAMES = [
    "setosa",
    "versicolor",
    "virginica"
]

def load_data():
    """Load the Iris dataset and format it as a DataFrame"""
    # Load the Iris dataset
    iris = load_iris()
    
    # Create a DataFrame for the features
    X = pd.DataFrame(iris.data, columns=FEATURE_NAMES)
    
    # Create a Series for the target
    y = pd.Series(iris.target, name="species")
    
    return X, y

def train_and_save_model():
    """Train a decision tree model on the Iris dataset and save it with MLflow"""
    # Set MLflow tracking URI to point to the MLflow server
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    
    # Set up MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load the data
    X, y = load_data()
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    # Start an MLflow run
    with mlflow.start_run() as run:
        # Create and train the model
        dt_params = {
            "criterion": "gini",
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        }
        
        dt_classifier = DecisionTreeClassifier(
            **dt_params,
            random_state=RANDOM_STATE
        )
        dt_classifier.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = dt_classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log parameters
        for param_name, param_value in dt_params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)
        
        # Plot and log the decision tree
        plt.figure(figsize=(15, 10))
        plot_tree(dt_classifier, 
                  feature_names=FEATURE_NAMES,
                  class_names=CLASS_NAMES,
                  filled=True, 
                  rounded=True)
        plt.savefig("plots/decision_tree.png")
        mlflow.log_artifact("plots/decision_tree.png")
        plt.close()
        
        # Plot and log the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig("plots/confusion_matrix.png")
        mlflow.log_artifact("plots/confusion_matrix.png")
        plt.close()
        
        # Log the model
        mlflow.sklearn.log_model(
            dt_classifier, 
            "model",
            registered_model_name=MODEL_NAME
        )
        
        # Print results
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print(f"Model saved with run ID: {run.info.run_id}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
        
        # Return the run ID and test data for later use
        return run.info.run_id, X_test, y_test

def load_and_evaluate_model(run_id=None):
    """Load the model from MLflow and evaluate it"""
    print("\n--- Loading and evaluating model ---")
    
    if run_id:
        # Load model from specific run
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        print(f"Loaded model from run: {run_id}")
    else:
        # Load the latest model from the registry
        try:
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
            print(f"Loaded latest model: {MODEL_NAME}")
        except Exception as e:
            print(f"Error loading model from registry: {str(e)}")
            print("Loading model from most recent run instead...")
            # Get the most recent run
            runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
            if len(runs) == 0:
                raise Exception("No runs found. Please train a model first.")
            
            most_recent_run = runs.iloc[0]
            run_id = most_recent_run.run_id
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            print(f"Loaded model from run: {run_id}")
    
    # Load test data
    X, y = load_data()
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print metrics
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    return model, X_test, y_test

def predict_sample(model, sample_data=None):
    """Make predictions on sample data"""
    print("\n--- Making predictions on sample data ---")
    
    if sample_data is None:
        # Create sample data if none provided
        samples = [
            [5.1, 3.5, 1.4, 0.2],  # Typical setosa
            [6.0, 2.7, 4.1, 1.0],  # Typical versicolor
            [7.2, 3.0, 5.8, 1.8]   # Typical virginica
        ]
        sample_data = pd.DataFrame(samples, columns=FEATURE_NAMES)
    
    # Make predictions
    predictions = model.predict(sample_data)
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(sample_data)
    else:
        probabilities = None
    
    # Print results
    print("\nSample Predictions:")
    for i, (_, row) in enumerate(sample_data.iterrows()):
        print(f"Sample {i+1}:")
        print(f"  Measurements: {row.tolist()}")
        print(f"  Predicted class: {CLASS_NAMES[predictions[i]]}")
        
        if probabilities is not None:
            prob = probabilities[i]
            print(f"  Confidence: {prob[predictions[i]]:.4f}")
            print(f"  Class probabilities: {', '.join([f'{CLASS_NAMES[j]}: {p:.4f}' for j, p in enumerate(prob)])}")
        print()
    
    return predictions

if __name__ == "__main__":
    # Train and save the model
    print("=== Training and saving model ===")
    run_id, X_test, y_test = train_and_save_model()
    
    # Load and evaluate the model
    model, _, _ = load_and_evaluate_model(run_id)
    
    # Make predictions on sample data
    predict_sample(model)
