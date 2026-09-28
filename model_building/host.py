# ---------------------------------------------------------------------------
# Hosting Script
# Creates a Hugging Face Space and pushes the deployment files
# (Dockerfile, app.py, requirements.txt) to it. HF Spaces reads the
# Dockerfile, builds the container, and serves the Streamlit app.
# Runs as the FOURTH job in the GitHub Actions pipeline.
# ---------------------------------------------------------------------------
from huggingface_hub import HfApi, login
import os

login(token=os.environ["HF_TOKEN"])

api = HfApi()

# Create the Space (type "docker" tells HF to build from Dockerfile)
api.create_repo(
    repo_id="NiiOsa1/tourism-package-prediction-app",
    repo_type="space",
    space_sdk="docker",
    exist_ok=True
)

# Push deployment files to the Space
deployment_dir = "tourism_project/deployment"
for filename in ["Dockerfile", "app.py", "requirements.txt"]:
    filepath = os.path.join(deployment_dir, filename)
    api.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=filename,
        repo_id="NiiOsa1/tourism-package-prediction-app",
        repo_type="space",
    )
    print(f"Uploaded: {filename}")

print("\nAll deployment files pushed to HF Space.")
print("URL: https://huggingface.co/spaces/NiiOsa1/tourism-package-prediction-app")
