# Iris Classification with MLflow

This project demonstrates a complete ML workflow using scikit-learn's Decision Tree classifier on the Iris dataset with MLflow for experiment tracking, model versioning, and deployment.

## Project Overview

The project trains a decision tree model that classifies iris flowers into one of three species (setosa, versicolor, or virginica) based on measurements of the flower's sepal and petal dimensions. It showcases:

1. **Training a model** with sklearn and tracking it using MLflow
2. **Model evaluation** with various metrics
3. **Saving and loading models** through MLflow's Model Registry
4. **Using the model for predictions** in a production-like environment
5. **Integration into a continuous workflow** in a simulated gardening company

## Project Structure

```
iris_mlflow_project/
│
├── requirements.txt       # Dependencies
├── iris_model.py          # Model training, evaluation, and prediction
└── gardening_example.py   # Example of continuous workflow
```

## Installation

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training and Evaluating the Model

```bash
python iris_model.py
```

This script:

- Loads the Iris dataset
- Trains a decision tree classifier
- Logs parameters, metrics, artifacts, and the model to MLflow
- Loads the model back from MLflow
- Evaluates the model using various metrics
- Makes predictions on sample iris measurements

### Simulating a Gardening Company Workflow

```bash
python gardening_example.py
```

This script simulates how the model could be used in a gardening company's continuous workflow for plant classification and care recommendations.

## MLflow Integration

This project uses MLflow for:

1. **Experiment Tracking**:

   - Track parameters, metrics, and artifacts
   - Log visualizations like decision tree plots and confusion matrices

2. **Model Registry**:
   - Save models with versioning
   - Load models for predictions

## Gardening Company Use Case

The `gardening_example.py` script demonstrates how this model could be used in a real-world gardening company:

1. **Continuous Monitoring**:

   - Regular measurements of plant characteristics
   - Automatic classification of plant types

2. **Automated Care Recommendations**:

   - Generate specific care instructions based on plant type and health
   - Track plant health over time

3. **Model Performance Monitoring**:
   - Regularly monitor model confidence
   - Trigger retraining when needed

## Performance Metrics

The model achieves:

- Accuracy: ~97.8%
- Precision: ~97.9%
- Recall: ~97.8%
- F1 Score: ~97.8%

## Viewing MLflow UI

To view the MLflow tracking UI:

```bash
mlflow ui
```

Then open your browser to http://localhost:8080
