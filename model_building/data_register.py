# ---------------------------------------------------------------------------
# Data Registration Script
# Purpose: Load the raw CSV, validate that all expected columns are present,
# and push it to Hugging Face Hub as a versioned dataset.
# This runs as the FIRST job in the GitHub Actions pipeline.
# ---------------------------------------------------------------------------
import pandas as pd
from datasets import Dataset
from huggingface_hub import login
import os

# Authenticate with Hugging Face using the token stored in environment secrets
login(token=os.environ["HF_TOKEN"])

# Path to the raw dataset (relative to the repo root)
RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that every expected column is present before registering
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# Push to Hugging Face Hub as a versioned dataset
ds = Dataset.from_pandas(df)
ds.push_to_hub("NiiOsa1/tourism-package-prediction")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"Columns: {list(df.columns)}")
print("ProdTaken distribution:")
print(df["ProdTaken"].value_counts())
