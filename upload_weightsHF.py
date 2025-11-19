import os
from huggingface_hub import HfApi
import getpass

# --- CONFIGURATION ---
REPO_ID = "HamzaAmmar/chesshacks-model" # Or your username if you switched
LOCAL_FILE = "weights/best.pt"
REMOTE_FILENAME = "best.pt"

# SECURE: Ask for token at runtime OR get from environment
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    print("🔑 Enter your Hugging Face Write Token (hidden input):")
    HF_TOKEN = getpass.getpass()
# ---------------------

print(f"🚀 Uploading {LOCAL_FILE} to {REPO_ID}...")

api = HfApi(token=HF_TOKEN)

api.upload_file(
    path_or_fileobj=LOCAL_FILE,
    path_in_repo=REMOTE_FILENAME,
    repo_id=REPO_ID,
    repo_type="model"
)

print("✅ Upload complete!")