# ---------------------------------------------------------------------------
# Data Preparation Script
# Pulls the registered dataset from Hugging Face, applies cleaning based on
# audit findings, splits into train/test, saves locally, and pushes splits
# back to HF. Runs as the SECOND job in the GitHub Actions pipeline.
#
# Cleaning applied (from audit):
#   1. Drop "Unnamed: 0" (auto-generated junk index column)
#   2. Drop "CustomerID" (identifier, not a predictive feature)
#   3. Fix Gender typo: "Fe Male" -> "Female"
#   4. No imputation needed (zero missing values confirmed by audit)
#   5. No outlier removal (XGBoost handles outliers naturally)
# ---------------------------------------------------------------------------
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import login
import os

login(token=os.environ["HF_TOKEN"])

# Load from the central store, not a local copy
ds = load_dataset("NiiOsa1/tourism-package-prediction")
df = ds["train"].to_pandas()

# --- Cleaning ---
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True)
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Gender values after fix: {list(df['Gender'].unique())}")
print(f"Columns: {list(df.columns)}")

# --- Train/test split ---
# stratify preserves the 81/19 class ratio in both halves
target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save locally (GitHub Actions passes these as artifacts to the next job)
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"\nTrain set: {Xtrain.shape[0]} rows")
print(f"Test set: {Xtest.shape[0]} rows")
print(f"Train purchase rate: {ytrain.mean():.2%}")
print(f"Test purchase rate: {ytest.mean():.2%}")

# --- Push splits back to HF ---
# Stored in a separate dataset from the raw data because cleaned splits
# have different columns (no CustomerID, no Unnamed: 0, target separated).
# HF requires all splits in one dataset to share the same schema.
train_combined = pd.concat([Xtrain, ytrain], axis=1)
test_combined = pd.concat([Xtest, ytest], axis=1)

Dataset.from_pandas(train_combined).push_to_hub(
    "NiiOsa1/tourism-package-prediction-splits", split="train"
)
Dataset.from_pandas(test_combined).push_to_hub(
    "NiiOsa1/tourism-package-prediction-splits", split="test"
)

print("\nAll splits uploaded to Hugging Face.")
