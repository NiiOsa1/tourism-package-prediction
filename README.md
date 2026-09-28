# Tourism Package Purchase Prediction - MLOps Pipeline

End-to-end machine learning pipeline for predicting whether a customer
will purchase a tourism package. Built with XGBoost, MLflow, Streamlit,
Docker, and GitHub Actions.

## Project Structure

tourism_project/
├── .github/workflows/
│   └── pipeline.yml          # GitHub Actions CI/CD pipeline
├── data/
│   └── tourism.csv           # Raw dataset
├── model_building/
│   ├── data_register.py      # Push raw data to HF Datasets
│   ├── prep.py               # Clean, split, push splits to HF
│   ├── train.py              # Train XGBoost, log to MLflow, push model to HF
│   └── host.py               # Deploy Streamlit app to HF Spaces
├── deployment/
│   ├── Dockerfile            # Docker container configuration
│   ├── app.py                # Streamlit prediction app
│   └── requirements.txt      # App dependencies
└── README.md

## Pipeline

The GitHub Actions pipeline runs four sequential jobs on every push to main:

1. Data Registration - Pushes raw data to Hugging Face Datasets
2. Data Preparation - Cleans data, creates train/test splits, pushes to HF
3. Model Training - Trains XGBoost with GridSearchCV, logs to MLflow, pushes model to HF
4. Deployment - Pushes Docker and Streamlit files to HF Spaces

## Live App

Streamlit App: https://huggingface.co/spaces/NiiOsa1/tourism-package-prediction-app

## Hugging Face Resources

- Dataset (raw): https://huggingface.co/datasets/NiiOsa1/tourism-package-prediction
- Dataset (splits): https://huggingface.co/datasets/NiiOsa1/tourism-package-prediction-splits
- Model: https://huggingface.co/NiiOsa1/tourism-package-model
- App: https://huggingface.co/spaces/NiiOsa1/tourism-package-prediction-app

## Model Performance

| Metric    | Train | Test |
|-----------|-------|------|
| Accuracy  | 1.00  | 0.94 |
| Precision | 1.00  | 0.85 |
| Recall    | 1.00  | 0.83 |
| F1 Score  | 1.00  | 0.84 |

## Tech Stack

- Model: XGBoost with scikit-learn pipeline
- Experiment Tracking: MLflow
- App Framework: Streamlit
- Containerization: Docker
- CI/CD: GitHub Actions
- Hosting: Hugging Face Spaces
