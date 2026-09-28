# ---------------------------------------------------------------------------
# Model Training Script
# Loads prepared train/test splits, builds a preprocessing + XGBoost pipeline,
# tunes hyperparameters via GridSearchCV, logs experiments to MLflow, evaluates
# the best model, and saves it for deployment.
# Runs as the THIRD job in the GitHub Actions pipeline.
# ---------------------------------------------------------------------------
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import mlflow
import os
from huggingface_hub import HfApi, login

login(token=os.environ["HF_TOKEN"])

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-prediction")

# Load prepared splits (passed as artifacts from the data-prep job)
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# Define feature groups
numeric_features = [
    "Age", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome", "CityTier"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation"
]

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Build preprocessing + model pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter grid (final round, coarse-to-fine tuned)
param_grid = {
    "xgbclassifier__n_estimators": [150, 200, 250],
    "xgbclassifier__max_depth": [6, 7, 8],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.8, 1.0],
    "xgbclassifier__colsample_bylevel": [0.5, 0.7],
    "xgbclassifier__reg_lambda": [1, 3],
}

with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring="f1"
    )
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations tested
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_f1", results["std_test_score"][i])

    mlflow.log_params(grid_search.best_params_)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

    # Evaluate best model
    best_model = grid_search.best_estimator_
    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"],
    })

    print("\nTRAIN SET:")
    print(classification_report(ytrain, y_pred_train))
    print("TEST SET:")
    print(classification_report(ytest, y_pred_test))

    # Save model locally for deployment
    model_path = "tourism_project/deployment/best_model.joblib"
    os.makedirs("tourism_project/deployment", exist_ok=True)
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")

    # Push model to Hugging Face Model Hub
    api = HfApi()
    api.create_repo(
        repo_id="NiiOsa1/tourism-package-model",
        repo_type="model",
        exist_ok=True
    )
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_model.joblib",
        repo_id="NiiOsa1/tourism-package-model",
        repo_type="model",
    )
    print("Model uploaded to Hugging Face Model Hub.")
